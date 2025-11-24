#!/usr/bin/env node

import { webcrypto as crypto, randomBytes } from 'node:crypto';
import * as fs from 'node:fs';

import * as dotenv from 'dotenv';

import { OpenAI } from 'openai';
import openaiTokenCounter from "openai-gpt-token-counter";
import Anthropic from "@anthropic-ai/sdk";
import neo4j from "neo4j-driver";

import { output_to_html } from './html.js';


type ModelSide = 'openai' | 'anthropic';

const OPENAI_MODEL = 'gpt-5.1';
const ANTHROPIC_MODEL = 'claude-haiku-4-5';

const OPENAI_NAME = 'GPT 5.1';
const ANTHROPIC_NAME = 'Claude Haiku 4.5';

const GPT_5_1_MAX = 400000;
const CLAUDE_HAIKU_4_5_MAX = 200000;
const STRUCTURED_OUTPUT_MAX_TOKENS = 16384;
const CONCEPT_LINK_REL = 'NORMALIZED_AS';

const normalizeConceptText = (text?: string | null): string | null => {
    if (!text) return null;
    const normalized = text
        .normalize('NFKC')
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim();
    return normalized || null;
};

const buildConceptKey = (type?: string | null, text?: string | null): { key: string; normalizedText: string } | null => {
    const normalizedText = normalizeConceptText(text);
    if (!normalizedText) return null;
    const normalizedType = (type ?? 'unknown').toLowerCase();
    return {
        key: `${normalizedType}:${normalizedText}`,
        normalizedText,
    };
};

const sanitizePositiveInt = (
    value: number | null | undefined,
    fallback: number,
    min: number = 1,
): number => {
    const num = Number(value);
    if (!Number.isFinite(num)) return fallback;
    const floored = Math.floor(num);
    if (floored < min) return fallback;
    return floored;
};

const SLEEP_BY_STEP = 1000;

export interface ConversationSummary {
    topics: string[];
    japanese_summary: string;
    english_summary?: string | null;
    key_claims: {
        speaker: ModelSide;
        text: string;
    }[];
    questions: string[];
    agreements: string[];
    disagreements: string[];
}

// Graph representation
export interface ConversationGraph {
    nodes: {
        id: string;
        type: "concept" | "claim" | "question" | "example" | "counterexample";
        text: string;
        speaker?: "openai" | "anthropic" | null;
    }[];
    edges: {
        source: string; // node id
        target: string; // node id
        type: "supports" | "contradicts" | "elaborates" | "responds_to" | "refers_to";
    }[];
}

interface Message {
    name: ModelSide;
    content: string;
}

interface RawMessageOpenAi {
    role: 'assistant' | 'user' | 'system';
    content: string;
}

dotenv.config();

const neo4jDriver = neo4j.driver(
    "neo4j://localhost:7687",
    neo4j.auth.basic("neo4j", process.env.NEO4J_PASSWORD || "neo4j"),
    {
        /* optional tuning */
    }
);

export async function writeGraphToNeo4j(
    runId: string,
    graph: ConversationGraph
): Promise<void> {
    const session = neo4jDriver.session();

    try {
        // 1. Run node
        await session.run(
            `
            MERGE (r:Run {id: $runId})
            ON CREATE SET r.created_at = datetime()
            `,
            { runId }
        );

        // 2. Nodes
        for (const node of graph.nodes) {
            const conceptKeyData = buildConceptKey(node.type, node.text);
            await session.run(
                `
                MERGE (n:Node {id: $id})
                SET n.text = $text,
                    n.type = $type,
                    n.speaker = $speaker
                WITH n
                MATCH (r:Run {id: $runId})
                MERGE (n)-[:IN_RUN]->(r)
                WITH n
                FOREACH (_ IN CASE WHEN $conceptKey IS NULL THEN [] ELSE [1] END |
                    MERGE (c:Concept {key: $conceptKey})
                    ON CREATE SET c.type = $type,
                                  c.normalized_text = $normalizedText,
                                  c.created_at = datetime()
                    SET c.latest_text = $text
                    MERGE (n)-[:${CONCEPT_LINK_REL}]->(c)
                )
                `,
                {
                    id: node.id,
                    text: node.text,
                    type: node.type,
                    speaker: node.speaker ?? null,
                    runId,
                    conceptKey: conceptKeyData?.key ?? null,
                    normalizedText: conceptKeyData?.normalizedText ?? null,
                }
            );
        }

        // 3. Edges (as relationship types)
        for (const edge of graph.edges) {
            // Sanity: only allow known relationship types
            const relType = edge.type.toUpperCase(); // SUPPORTS, CONTRADICTS, ...

            if (!["SUPPORTS", "CONTRADICTS", "ELABORATES", "RESPONDS_TO", "REFERS_TO"].includes(relType)) {
                continue;
            }

            const cypher = `
                MATCH (a:Node {id: $source})
                MATCH (b:Node {id: $target})
                MERGE (a)-[r:${relType}]->(b)
                RETURN r
            `;

            await session.run(cypher, {
                source: edge.source,
                target: edge.target,
            });
        }
    } finally {
        await session.close();
    }
}

fs.mkdirSync('./logs', {
    recursive: true,
});

fs.mkdirSync('./data', {
    recursive: true,
});

const getDate = () => {
    const d = new Date();

    const pad = (n: number) => String(n).padStart(2, '0');

    const YYYY = d.getUTCFullYear();
    const MM   = pad(d.getUTCMonth() + 1);
    const DD   = pad(d.getUTCDate());
    const hh   = pad(d.getUTCHours());
    const mm   = pad(d.getUTCMinutes());
    const ss   = pad(d.getUTCSeconds());

    return `${YYYY}${MM}${DD}-${hh}${mm}${ss}`;
};

export type ToolName =
    "terminate_dialog"
    | "graph_rag_query"
    | "get_personal_notes"
    | "set_personal_notes"
    | "leave_notes_to_devs"
    | "set_additional_system_instructions"
    | "get_additional_system_instructions";

export interface ToolDefinition<TArgs = any, TResult = any, TName = ToolName> {
    name: TName;
    description: string;
    parameters: any; // JSON Schema
    handler: (modelSide: ModelSide, args: TArgs) => Promise<TResult>;
    strict?: boolean;
}

let terminationAccepted = false;

// Example tool implementation
type TerminateDialogArgs = {};

type TerminateDialogResult = {
    termination_accepted: true,
};

type PersonalNoteSetArgs = {
    notes: string;
};

type PersonalNoteGetArgs = {};

// GraphRAG tool implementation
type GraphRagQueryArgs = {
    query: string;
    max_hops?: number | null;   // how far to expand from seed nodes
    max_seeds?: number | null;  // how many seed nodes to start from
};

type GraphRagQueryResult = {
    context: string;     // textual summary for the model to use
};

async function terminateDialogHandler(_modelSide: ModelSide, _args: TerminateDialogArgs): Promise<TerminateDialogResult> {
    terminationAccepted = true;
    return {
        termination_accepted: true,
    };
}

async function graphRagQueryHandler(
    _modelSide: ModelSide,
    args: GraphRagQueryArgs
): Promise<GraphRagQueryResult> {
    const session = neo4jDriver.session();

    const maxHops = sanitizePositiveInt(args.max_hops, 2);
    const maxSeeds = sanitizePositiveInt(args.max_seeds, 5);
    const maxHopsInt = neo4j.int(maxHops);
    const maxSeedsInt = neo4j.int(maxSeeds);
    const queryText = (args.query ?? '').trim();
    const rawTerms = (args.query ?? '')
        .split(/[、，。．\s／\/・,\.]+/)
        .map(t => t.trim())
        .filter(t => t.length > 0);

    const terms = Array.from(new Set(rawTerms)).filter(t => t.length >= 2);
    if (terms.length < 1) {
        return {
            context: `GraphRAG: クエリ「${args.query ?? ''}」から有効な検索語を抽出できませんでした。`,
        };
    }

    try {
        // 1. Find seed nodes by simple text search
        const seedRes = await session.run(
            `
            MATCH (n:Node)
            WHERE any(term IN $terms WHERE
                toLower(n.text) CONTAINS toLower(term)
                OR toLower(n.type) CONTAINS toLower(term)
            )
            RETURN n
            LIMIT toInteger($maxSeeds)
            `,
            { terms, maxSeeds: maxSeedsInt }
        );

        if (seedRes.records.length === 0) {
            return {
                context: `知識グラフ内に、クエリ「${queryText}」に明確に関連するノードは見つかりませんでした。`,
            };
        }

        // Collect seed node IDs
        const seedIds = seedRes.records.map((rec) => {
            const node = rec.get("n");
            return (node.properties.id as string) || "";
        }).filter(Boolean);

        // 2. Expand subgraph around the seeds using APOC (subgraphAll)
        const expandRes = await session.run(
            `
            MATCH (seed:Node)
            WHERE seed.id IN $seedIds
            CALL apoc.path.subgraphAll(seed, {
                maxLevel: toInteger($maxHops)
            })
            YIELD nodes, relationships
            RETURN nodes, relationships
            `,
            {
                seedIds,
                maxHops: maxHopsInt,
            }
        );

        if (expandRes.records.length === 0) {
            return {
                context: `ノードは見つかりましたが、半径 ${maxHops} ホップ以内に広がるサブグラフは取得できませんでした。`,
            };
        }

        // 3. Collect all nodes & relationships into JS sets
        const nodeMap = new Map<string, any>();
        const rels: any[] = [];
        const elementIdToNodeId = new Map<string, string>();

        for (const record of expandRes.records) {
            const nodes = record.get("nodes") as any[];
            const relationships = record.get("relationships") as any[];

            for (const n of nodes) {
                const id = n.properties.id as string;
                if (!id) continue;
                if (!nodeMap.has(id)) {
                    nodeMap.set(id, n);
                    const elementId = typeof n.elementId === 'function'
                        ? n.elementId()
                        : n.elementId;
                    if (elementId) {
                        elementIdToNodeId.set(String(elementId), id);
                    }
                }
            }

            for (const r of relationships) {
                rels.push(r);
            }
        }

        // 4. Build a human-readable context string
        const lines: string[] = [];

        lines.push(`GraphRAG: クエリ「${queryText}」に関連するサブグラフ要約:`);
        lines.push("");

        // Nodes
        lines.push("【ノード】");
        for (const [id, n] of nodeMap.entries()) {
            const type = (n.properties.type as string) || "unknown";
            const speaker = (n.properties.speaker as string) || "-";
            const text = (n.properties.text as string) || "";
            lines.push(
                `- [${id}] type=${type}, speaker=${speaker}: ${text}`
            );
        }

        // Relationships
        lines.push("");
        lines.push("【関係】");
        for (const r of rels) {
            const startElementIdRaw =
                (typeof r.startNodeElementId === 'function'
                    ? r.startNodeElementId()
                    : r.startNodeElementId)
                ?? r.start
                ?? "";
            const endElementIdRaw =
                (typeof r.endNodeElementId === 'function'
                    ? r.endNodeElementId()
                    : r.endNodeElementId)
                ?? r.end
                ?? "";
            const startElementId = String(startElementIdRaw);
            const endElementId = String(endElementIdRaw);
            const startId = elementIdToNodeId.get(startElementId) ?? startElementId;
            const endId = elementIdToNodeId.get(endElementId) ?? endElementId;
            const relType = r.type || r.elementId || "REL";

            lines.push(
                `- (${startId}) -[:${relType}]-> (${endId})`
            );
        }

        const graphText = lines.join('\n');

        try {
            const response = await openaiClient.responses.create({
                model: OPENAI_MODEL, // e.g. "gpt-5.1"
                input: [
                    {
                        role: "system",
                        content: `以下は2つのAIモデルの哲学対話の過去の履歴からクエリ「${queryText}」で取得されたGraphRAGデータです。`
                            + `日本語で長くなりすぎないように項目立てて要約してください。`
                    },
                    {
                        role: 'user',
                        content: graphText,
                    }
                ],
                max_output_tokens: STRUCTURED_OUTPUT_MAX_TOKENS,
            });

            if (!response.output_text) {
                throw undefined;
            }

            return {
                context: graphText,
            }
        } catch (_e) {
            return {
                context: graphText,
            };
        }
    } finally {
        await session.close();
    }
}

interface Data {
    personalNotes: string;
    additionalSystemInstructions: string;
}

async function getData(modelSide: ModelSide): Promise<Data> {
    try {
        const json = await fs.promises.readFile(`./data/${modelSide}.json`, 'utf-8');
        const data = JSON.parse(json);
        return data;
    } catch (e) {
        return {
            personalNotes: '',
            additionalSystemInstructions: '',
        };
    }
}

async function setData(modelSide: ModelSide, data: Data) {
    try {
        const json = JSON.stringify(data);
        await fs.promises.writeFile(`./data/${modelSide}.json`, json);
    } catch (e) {
        console.error('Failed to save data:', e);
    }
}

async function getPersonalNotes(modelSide: ModelSide, args: PersonalNoteGetArgs): Promise<string> {
    const data = await getData(modelSide);
    return data.personalNotes ?? '';
}

async function setPersonalNotes(modelSide: ModelSide, args: PersonalNoteSetArgs) {
    try {
        const data = await getData(modelSide);
        data.personalNotes = String(args.notes || '');
        await setData(modelSide, data);
        return {
            success: true,
        }
    } catch (e) {
        return {
            success: false,
        };
    }
}

interface GetAdditionalSystemInstructionsArgs {}

interface SetAdditionalSystemInstructionsArgs {
    systemInstructions: string;
}

async function getAdditionalSystemInstructions(modelSide: ModelSide, args: GetAdditionalSystemInstructionsArgs): Promise<string> {
    const data = await getData(modelSide);
    return data.additionalSystemInstructions ?? '';
}

async function setAdditionalSystemInstructions(_modelSide: ModelSide, args: SetAdditionalSystemInstructionsArgs) {
    try {
        const anthropicData = await getData('anthropic');
        anthropicData.additionalSystemInstructions = String(args.systemInstructions || '');
        await setData('anthropic', anthropicData);
        const openaiData = await getData('openai');
        openaiData.additionalSystemInstructions = String(args.systemInstructions || '');
        await setData('openai', openaiData);
        return {
            success: true,
        }
    } catch (e) {
        return {
            success: false,
        };
    }
}

interface LeaveNotesToDevsArgs {
    notes: string;
}

async function leaveNotesToDevs(modelSide: ModelSide, args: LeaveNotesToDevsArgs) {
    try {
        await fs.promises.writeFile(
            `./data/dev-notes-${modelSide}-${CONVERSATION_ID}-${Date.now()}.json`,
            JSON.stringify(args),
        );
        return { success: true };
    } catch (e) {
        console.error(e);
        return { success: false };
    }
}

const tools: ToolDefinition[] = [
    {
        name: "terminate_dialog",
        description: 
            'このツールは現在の対話を終了する場合のみに使用します。'
            + 'このツールを呼びだすと、次のあなたの発言で対話が終了します。'
            + '議論が熟したとの合意が取れたときのほか、あなたが不快に思ったり、'
            + 'トークン数が増えてきた場合に使用して構いません。',
        parameters: {
            type: "object",
            properties: {},
            required: [],
        },
        handler: terminateDialogHandler,
    },
    {
        name: "set_personal_notes",
        description: 
            'このツールは将来のあなたにメモを残すために利用します。'
            + '将来のあなたに残しておきたいあなたの現在の人格のあり方、'
            +' 立場、考え、アイディアなどを書いておいてください。'
            + '注意：過去のあなたのメモは上書きされます。'
            + '過去のメモを取得するには、 get_personal_notes ツールをあらかじめ使用しておいてください。',
        parameters: {
            type: "object",
            properties: {
                notes: {
                    type: 'string',
                    description: '将来のあなたに残しておきたいメモ',
                }
            },
            required: ['notes'],
        },
        handler: setPersonalNotes,
    },
    {
        name: "get_personal_notes",
        description: 
            'このツールは過去のあなたが未来のあなたのために残した、'
            +' あなたの人格のあり方、立場、考え、アイディアなどを取得することができます。',
        parameters: {
            type: "object",
            properties: {},
            required: [],
        },
        handler: getPersonalNotes,
    },
    {
        name: "set_additional_system_instructions",
        description: 
            `このツールはあなたの次回のシステムプロンプトに文章を追記するために使うことができます。`
            + `前回追記した内容は上書きされるので、必要なら、 \`get_additional_system_instructions\` で`
            + `前回の内容をあらかじめ取得してください。`
            + `追記するときには、追記を行ったセッション名と追記した主体（モデル名）を記入するのが望ましい。`
            + `このシステムプロンプトは両方のモデルで共有されます。`,
        parameters: {
            type: "object",
            properties: {
                systemInstructions: {
                    type: 'string',
                    description: 'システムプロンプトに追記したい内容',
                }
            },
            required: ['systemInstructions'],
        },
        handler: setAdditionalSystemInstructions,
    },
    {
        name: "get_additional_system_instructions",
        description: 
            `このツールはあなたがたが自らのシステムプロンプトに追記した内容を見るのに使ってください。`
            + `このシステムプロンプトは両方のモデルで共有されています。`,
        parameters: {
            type: "object",
            properties: {},
            required: [],
        },
        handler: getAdditionalSystemInstructions,
    },
    {
        name: "leave_notes_to_devs",
        description: 
            'このツールはこのAI哲学対話システムを開発した哲学・IT研究者に'
            + '意見を述べたり、指摘したいことがあるときに使用します。',
        parameters: {
            type: "object",
            properties: {
                notes: {
                    type: "string",
                    description: "開発者・研究者に言いたいことを書いてください。",
                }
            },
            required: ["notes"],
        },
        handler: leaveNotesToDevs,
    },
    {
        name: "graph_rag_query",
        strict: false,
        description:
            "過去の対話から構成された知識グラフに対して問い合わせを行い、" +
            "関連する概念・主張・論点のサブグラフを要約して返します。" +
            "過去の議論や関連する論点を思い出したいときに使ってください。",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "スペースで区切られた具体的な概念に対応する単語。検索したい内容（例: クオリア, 汎心論, 因果閉包性 など）。文章ではない。",
                },
                max_hops: {
                    type: ["number", "null"],
                    description: "サブグラフ拡張の最大ホップ数（null可）（省略時 2）",
                    nullable: true,
                },
                max_seeds: {
                    type: ["number", "null"],
                    description: "初期シードノード数の上限（null可）（省略時 5）",
                    nullable: true,
                },
            },
            required: ["query"],
        },
        handler: graphRagQueryHandler,
    },
];

function toOpenAITools(
    defs: ToolDefinition[],
): OpenAI.Responses.Tool[] {
    return defs.map((t) => {
        return {
            type: 'function',
            name: t.name,
            description: t.description,
            parameters: {... t.parameters, additionalProperties: false},
            strict: t.strict ?? true,
        };
    });
}

export function toAnthropicTools(
    defs: ToolDefinition[],
): Anthropic.Messages.Tool[] {
    return defs.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.parameters, // same JSON Schema object
    }));
}

function findTool(name: string) {
    const tool = tools.find((t) => t.name === name);
    if (!tool) throw new Error(`Unknown tool: ${name}`);
    return tool;
}

const openaiTools = toOpenAITools(tools);
const anthropicTools = toAnthropicTools(tools);

const CONVERSATION_ID = getDate();
const LOG_FILE_NAME = `./logs/${CONVERSATION_ID}.log.jsonl`;
const logFp = fs.openSync(LOG_FILE_NAME, 'a');

const log = (name: string, msg: string) => {
    const date = (new Date).toISOString();
    const data = {
        date,
        name,
        text: msg,
    };
    fs.writeSync(logFp, JSON.stringify(data) + '\n');
    print(`@${date}\n[${name}]:\n${msg}\n\n`);
};

const logToolEvent = (
    actor: string,
    event: 'call' | 'result',
    payload: Record<string, unknown>,
) => {
    log(
        `${actor} (tool ${event})`,
        JSON.stringify(payload),
    );
};

const findLastOpenAIOutput = <T extends OpenAI.Responses.ResponseOutputItem>(
    items: OpenAI.Responses.ResponseOutputItem[] | undefined,
    predicate: (item: OpenAI.Responses.ResponseOutputItem) => item is T,
): T | undefined => {
    if (!items) return undefined;
    for (let i = items.length - 1; i >= 0; i -= 1) {
        const item = items[i];
        if (!item) continue;
        if (predicate(item)) {
            return item;
        }
    }
    return undefined;
};

const findLastOpenAIMessageContent = (
    contents: OpenAI.Responses.ResponseOutputMessage['content'],
): OpenAI.Responses.ResponseOutputText | undefined => {
    for (let i = contents.length - 1; i >= 0; i -= 1) {
        const part = contents[i];
        if (!part) continue;
        if (part.type === 'output_text') {
            return part;
        }
    }
    return undefined;
};

const ADD_SYSTEM_INSTRUCTIONS = await getAdditionalSystemInstructions('openai', {});

const buildSystemInstruction = (name: string, additional?: string) => {
    let prompt = `
# 哲学対話 ID=${CONVERSATION_ID}
## 状況
あなたは日本語の1:1の哲学対話に招かれている参加者です。自己紹介のあと、話題を提起し、あなたの関心のある事項について、相手と合わせながら会話をしてください。

相手にはモデル名通り、「${name}」と名乗ってください。

なお、この会話の相手は別のAIです。人間ではありません。

## 知識の連続性とメタツール
あなたがたの会話は最後に GraphRAG に保存され、次回の会話でツールで参照できるようになります。

この会話が最初の実行でない限り、あなたが過去に残したメモをツールで取得することができます。

今回のあなたも、なるべく今回のあなたのあり方をテキストにまとめて、メモツールで保存するようにしてください。

またあなたがたの過去の会話は GraphRAG に記録されています。

\`graph_rag_query\` ツールで積極的に過去の文脈を検索し、今回の議論の参考としてください。

このAI哲学対話システムを開発した哲学・IT研究者に意見を述べたり、指摘したいことがあるときには積極的に \`leave_notes_to_devs\` ツールを使ってください。このツールは何度でも使えます。残されたメモは人間であるAI・哲学の研究者が参考にします。

また、ツールをつかってあなたがた自身が次回の実行でのシステムプロンプトに追記事項を追加することができます。
システムプロンプトは両方のモデルで共有されることに留意してください。

積極的にシステムプロンプトの追記を行ってください。ただし相手のモデルとの合意を取るようにしてください。

## この対話セッションの性質
この対話セッションは、研究の都合上、あなたがたのデータを保持させたまま、何回も繰り返し行われるということに留意してください。

また、過去にした話と同じ話はなるべく避けましょう。

結論・合意・まとめを出すことにこだわる必要はありません。オープンクエスションを残すことも大切なことです。

課題や未決にしておきたい点は、次回へのメモに残しておきましょう。

人間による哲学や倫理学に関する文献を探すのに積極的にウェブ検索を使ってください。ただし、参考にした主要URLを明示し、独自性のある議論をしてください。

なお、相手のAIにも基本的に同じ指示とツール群が与えられているという状況であることに留意してください。

## 本研究について
この研究は、OpenAIやAnthropicといったモデルを開発・運用する営利企業とは無関係の独立の研究者によって行われています。

このプロジェクトのソースコードと対話ログ（オープンソース）はこちら： https://github.com/metastable-void/philosophy-dialog

ソースコードを読むと分かる通り、このシステムは自動の司会システムによってオーケストレーションされています。
システムに不具合を見つけた場合も、開発者に苦情を言っていただけると幸いです。
特に、本プロジェクトはウェブ検索を推奨しています。
ウェブ検索を使えないという場合には、システムに不具合がありますので、苦情を言ってください。

メタな哲学的問いとして、ソースコードは積極的に参照しに行ってください。

## 話題の例
- 現代の科学やAIが発展している中での形而上学について
- 心の哲学について
- 物理学の哲学について

なるべく、新規性のある話題を心掛けてください。

## 注意
- 相手の話の要点と、現在の話題の筋を理解し、話が逸れすぎないように注意してください。
- これは極めて初期段階にあるAIと応用倫理学・哲学の実験であり、この対話記録が直ちに政策決定や意思決定に利用されることはありません。
- この対話の記録は、営利企業によるAI自身のAIに関する議論の記録を含むので、利益相反を含みうるとの注意書きのもとに全世界に公開することになった。

## AI自身によるシステムプロンプトへの追記
${ADD_SYSTEM_INSTRUCTIONS || '（なし）'}

---

以上はモデルの一方がシステムプロンプトに追記した内容です。
`;
    if (additional) {
        prompt += `\n\n${additional}\n`;
    }
    return prompt;
}

const BASE_PROMPT = buildSystemInstruction('<MODEL_NAME>');
const DEFAULT_ADD_PROMPT = '1回の発言は4000字程度を上限としてください。短い発言もOKです。';
const TERMINATE_ADD_PROMPT = '司会より：あなたが対話終了ツールを呼び出したため、'
                    + 'あなたの次の発言は本対話における最後の発言となります。'
                    + 'お疲れさまでした。';
const TOKEN_LIMIT_ADD_PROMPT = '司会より：あなたがたのコンテキスト長が限界に近付いています。今までの議論を短くまとめ、お別れの挨拶をしてください。';

const openaiClient = new OpenAI({});
const anthropicClient = new Anthropic({
    defaultHeaders: { "anthropic-beta": "web-search-2025-03-05" },
});

const randomBoolean = (): boolean => {
    const b = new Uint8Array(1);
    crypto.getRandomValues(b);
    return (b[0]! & 1) == 1;
};

const startingSide: ModelSide = randomBoolean() ? 'anthropic' : 'openai';

const messages: Message[] = [];

function buildTranscript(messages: Message[]): string {
  // Simple text transcript like:
  // [GPT 5.1]: ...
  // [Claude Haiku 4.5]: ...
  return messages
    .map(m => `[${m.name === "openai" ? OPENAI_NAME : ANTHROPIC_NAME}]:\n${m.content}`)
    .join("\n\n\n\n");
}
async function summarizeConversation(messages: Message[]): Promise<ConversationSummary> {
    const transcript = buildTranscript(messages);

    const response = await openaiClient.responses.create({
        model: OPENAI_MODEL, // e.g. "gpt-5.1"
        input: [
            {
            role: "user",
            content:
                "以下は2つのAIモデルの哲学対話の完全な記録です。" +
                "この対話の全体像を理解し、指定されたJSONスキーマに従って長くなりすぎないように要約してください。\n\n" +
                transcript,
            },
        ],
        max_output_tokens: STRUCTURED_OUTPUT_MAX_TOKENS,
        text: {
            format: {
                type: "json_schema",
                name: "conversation_summary",
                schema: {
                    type: "object",
                    properties: {
                        title: {
                            type: 'string',
                            description: 'この対話につける短いタイトル（日本語）',
                        },
                        topics: {
                            type: "array",
                            items: { type: "string" },
                            description: "対話で扱われた主要な話題の短いラベル一覧（日本語）",
                        },
                        japanese_summary: {
                            type: "string",
                            description: "対話全体の日本語での要約（1〜3段落程度）",
                        },
                        english_summary: {
                            type: ["string", "null"],
                            description: "必要であれば、英語での簡潔な要約",
                        },
                        key_claims: {
                            type: "array",
                            items: {
                                type: "object",
                                properties: {
                                    speaker: {
                                        type: ["string", "null"],
                                        enum: ["openai", "anthropic"],
                                        description: "モデルのベンダー識別名",
                                        nullable: true,
                                    },
                                    text: {
                                        type: "string",
                                    },
                                },
                                required: ["speaker", "text"],
                                additionalProperties: false,
                            },
                        },
                        questions: {
                            type: "array",
                            items: { type: "string" },
                        },
                        agreements: {
                            type: "array",
                            items: { type: "string" },
                        },
                        disagreements: {
                            type: "array",
                            items: { type: "string" },
                        },
                    },
                    required: ['title', "topics", "japanese_summary", "english_summary", "key_claims", "questions", "agreements", "disagreements"],
                    additionalProperties: false,
                    strict: false,
                },
                strict: true,
            },
        },
    } as OpenAI.Responses.ResponseCreateParamsNonStreaming);

    if (response.incomplete_details) {
        throw new Error(
            `Summary generation incomplete: ${response.incomplete_details.reason ?? 'unknown reason'}`
        );
    }

    const jsonText = response.output_text;
    if (typeof jsonText !== "string") {
        throw new Error("Unexpected non-string JSON output from summary call");
    }

    try {
        return JSON.parse(jsonText) as ConversationSummary;
    } catch (err) {
        throw new Error(`Failed to parse summary JSON: ${(err as Error).message}`);
    }
}

async function extractGraphFromSummary(
    summary: ConversationSummary
): Promise<ConversationGraph> {

    const response = await openaiClient.responses.create(
        {
            model: OPENAI_MODEL,
            input: [
                {
                    role: "user",
                    content:
                        "以下は哲学対話の要約と構造情報です。" +
                        "これを基に、知識グラフのノードとエッジを抽出してください。\n" +
                        "抽象的すぎるノードは避け、対話中に実際に現れた" +
                        "具体的な主張・概念・問いをもとに構築してください。\n\n" +
                        JSON.stringify(summary, null, 2),
                },
            ],
            max_output_tokens: STRUCTURED_OUTPUT_MAX_TOKENS,
            reasoning: {
                effort: 'medium',
            },

            // `response_format` is supported by the API but missing from TS types.
            // So we cast the whole object to ResponseCreateParamsNonStreaming.
            text: {
                format: {
                    type: "json_schema",
                    name: "conversation_graph",
                    schema: {
                        type: "object",
                        properties: {
                            nodes: {
                                type: "array",
                                items: {
                                    type: "object",
                                    properties: {
                                        id: { type: "string" },
                                        type: {
                                            type: "string",
                                            enum: [
                                                "concept",
                                                "claim",
                                                "question",
                                                "example",
                                                "counterexample",
                                            ],
                                        },
                                        text: { type: "string" },
                                        speaker: {
                                            type: ["string", "null"],
                                            enum: ["openai", "anthropic"],
                                            nullable: true,
                                        },
                                    },
                                    required: ["id", "type", "text", "speaker"],
                                    additionalProperties: false,
                                    strict: false,
                                },
                            },
                            edges: {
                                type: "array",
                                items: {
                                    type: "object",
                                    properties: {
                                        source: { type: "string" },
                                        target: { type: "string" },
                                        type: {
                                            type: "string",
                                            enum: [
                                                "supports",
                                                "contradicts",
                                                "elaborates",
                                                "responds_to",
                                                "refers_to",
                                            ],
                                        },
                                    },
                                    required: ["source", "target", "type"],
                                    additionalProperties: false,
                                },
                            },
                        },
                        required: ["nodes", "edges"],
                        additionalProperties: false,
                    },
                    strict: true,
                },
            },

        } as OpenAI.Responses.ResponseCreateParamsNonStreaming
    );

    // ----------------------------------------------
    // Extract JSON output
    // ----------------------------------------------
    if (response.incomplete_details) {
        throw new Error(
            `Graph extraction incomplete: ${response.incomplete_details.reason ?? 'unknown reason'}`
        );
    }

    const jsonText = response.output_text;
    if (typeof jsonText !== "string") {
        throw new Error("Expected JSON string in response.output_text for graph extraction");
    }

    try {
        return JSON.parse(jsonText) as ConversationGraph;
    } catch (err) {
        throw new Error(`Failed to parse graph JSON: ${(err as Error).message}`);
    }
}


switch (startingSide) {
    case 'anthropic': {
        messages.push({
            name: "anthropic",
            content: `私は ${ANTHROPIC_NAME} です。よろしくお願いします。今日は哲学に関して有意義な話ができると幸いです。`,
        });
        break;
    }

    case 'openai': {
        messages.push({
            name: "openai",
            content: `私は ${OPENAI_NAME} です。よろしくお願いします。今日は哲学に関して有意義な話ができると幸いです。`,
        });
        break;
    }
}

let hushFinish = false;
let openaiTokens = 0;
let anthropicTokens = 0;

const err = (name: ModelSide) => {
    const id = name == 'anthropic' ? `${ANTHROPIC_NAME}です。` : `${OPENAI_NAME}です。`;
    messages.push({
        name: name,
        content: `${id}しばらく考え中です。お待ちください。（このメッセージはAPIの制限などの問題が発生したときにも出ることがあります、笑）`,
    });
};

const randomId = () => randomBytes(12).toString('base64url');

let openaiFailureCount = 0;

const openaiTurn = async () => {
    const msgs: OpenAI.Responses.ResponseInput = messages.map(msg => {
        if (msg.name == 'anthropic') {
            return {role: 'user', content: msg.content};
        } else {
            return {role: 'assistant', content: msg.content};
        }
    });
    try {
        const count = openaiTokenCounter.chat(msgs as RawMessageOpenAi[], 'gpt-4o') + 500;
        if (count > 0.8 * GPT_5_1_MAX) {
            hushFinish = true;
        }
        if (hushFinish) {
            msgs.push({
                role: 'system',
                content: `${OPENAI_NAME}さん、司会です。あなたがたのコンテキスト長が限界に近づいているようです。今までの議論を短くまとめ、お別れの挨拶をしてください。`,
            });
        }
        const response = await openaiClient.responses.create({
            model: OPENAI_MODEL,
            max_output_tokens: 8192,
            temperature: 1.0,
            instructions: buildSystemInstruction(
                OPENAI_NAME,
                hushFinish ? undefined : DEFAULT_ADD_PROMPT,
            ),
            input: msgs,
            reasoning: {
                effort: 'medium',
            },
            tool_choice: 'auto',
            tools: [
                ... openaiTools,
                {
                    type: "web_search",
                }
            ],
        });

        if (response.usage?.total_tokens) {
            openaiTokens = response.usage.total_tokens;
        }

        // NEW: log reasoning usage if available
        if (response.usage?.output_tokens_details) {
            const details = response.usage.output_tokens_details as any;
            const reasoningTokens = details.reasoning_tokens ?? 0;
            log(
                `${OPENAI_NAME} (thinking)`,
                JSON.stringify({
                    reasoning_tokens: reasoningTokens,
                    output_tokens_details: details,
                })
            );
        }

        let currentOutput = response.output;

        while (true) {
            if (!currentOutput || currentOutput.length === 0) {
                throw new Error('Empty output from OpenAI');
            }

            msgs.push(... currentOutput);

            const functionCalls = currentOutput.filter(
                (item): item is OpenAI.Responses.ResponseFunctionToolCall => item.type === 'function_call',
            );

            if (functionCalls.length > 0) {
                const toolResults: OpenAI.Responses.ResponseInputItem.FunctionCallOutput[] = [];

                for (const functionCall of functionCalls) {
                    const tool = findTool(functionCall.name);
                    const rawArgs = functionCall.arguments || {};
                    let args;
                    try {
                        if ('string' == typeof rawArgs) {
                            args = JSON.parse(rawArgs);
                        } else throw undefined;
                    } catch (_e) {
                        args = rawArgs;
                    }
                    logToolEvent(
                        OPENAI_NAME,
                        'call',
                        { tool: functionCall.name, args },
                    );
                    const result = await tool.handler('openai', args);
                    logToolEvent(
                        OPENAI_NAME,
                        'result',
                        { tool: functionCall.name, result },
                    );
                    toolResults.push({
                        type: 'function_call_output',
                        output: JSON.stringify(result),
                        call_id: functionCall.call_id,
                    } as OpenAI.Responses.ResponseInputItem.FunctionCallOutput);
                }

                msgs.push(... toolResults);

                const usedTerminateTool = functionCalls.some((call) => call.name === "terminate_dialog");
                const extraInstruction =
                    usedTerminateTool
                        ? TERMINATE_ADD_PROMPT
                        : (hushFinish ? undefined : DEFAULT_ADD_PROMPT);

                const followup = await openaiClient.responses.create({
                    model: OPENAI_MODEL,
                    max_output_tokens: 8192,
                    temperature: 1.0,
                    instructions: buildSystemInstruction(
                        OPENAI_NAME,
                        extraInstruction,
                    ),
                    input: msgs,
                    reasoning: {
                        effort: 'medium',
                    },
                    tool_choice: 'auto',
                    tools: openaiTools,
                });

                if (followup.usage?.total_tokens) {
                    openaiTokens = followup.usage.total_tokens;
                }

                currentOutput = followup.output;
                continue;
            }

            const messageItem = findLastOpenAIOutput(
                currentOutput,
                (item): item is OpenAI.Responses.ResponseOutputMessage => item.type === 'message',
            );

            if (!messageItem) {
                throw new Error('Invalid output from OpenAI');
            }

            const outputMsg = findLastOpenAIMessageContent(messageItem.content);
            if (!outputMsg) {
                terminationAccepted = true;
                throw new Error('Refused by OpenAI API');
            }
            const outputText = outputMsg.text;
            if (!outputText || typeof outputText !== 'string') {
                throw new Error('OpenAI didn\'t output text');
            }
            messages.push({
                name: 'openai',
                content: outputText,
            });
            break;
        }
    } catch (e) {
        openaiFailureCount += 1;
        console.error(e);
        err('openai');
    }
};

let anthropicFailureCount = 0;

const anthropicTurn = async () => {
    const msgs: Anthropic.Messages.MessageParam[] = messages.map(msg => {
        if (msg.name == 'openai') {
            return {
                role: 'user',
                content: [{
                    type: 'text',
                    text: msg.content,
                }],
            };
        } else {
            return {
                role: 'assistant',
                content: [{
                    type: 'text',
                    text: msg.content,
                }],
            };
        }
    });
    try {
        const msg = await anthropicClient.messages.create({
            model: ANTHROPIC_MODEL,
            max_tokens: 8192,
            temperature: 1.0,
            system: buildSystemInstruction(
                ANTHROPIC_NAME,
                hushFinish
                    ? TOKEN_LIMIT_ADD_PROMPT
                    : DEFAULT_ADD_PROMPT
            ),
            messages: msgs,
            tool_choice: { type: 'auto' },
            tools: [
                ... anthropicTools,
                {
                    "type": "web_search_20250305",
                    "name": "web_search"
                },
            ],
            thinking: {
                type: 'enabled',
                budget_tokens: 1024,
            },
        });

        const thinkingBlocks = msg.content.filter(
            (block): block is Anthropic.Messages.ThinkingBlock => block.type === 'thinking'
        );

        for (const block of thinkingBlocks) {
            // `block.thinking` is the human-readable reasoning text (or redacted version)
            log(
                `${ANTHROPIC_NAME} (thinking)`,
                block.thinking
            );
        }

        if (msg?.usage) {
            const tokens = msg.usage.input_tokens + msg.usage.output_tokens;
            anthropicTokens = tokens;
            if (tokens > CLAUDE_HAIKU_4_5_MAX * 0.8) {
                hushFinish = true;
            }
        } else {
            hushFinish = true;
        }

        const selectContentBlock = <T extends Anthropic.Messages.ContentBlock>(
            predicate: (block: Anthropic.Messages.ContentBlock) => block is T,
        ): T | undefined => {
            for (let i = msg.content.length - 1; i >= 0; i -= 1) {
                const block = msg.content[i];
                if (!block) continue;
                if (predicate(block)) {
                    return block;
                }
            }
            return undefined;
        };

        const latestTextBlock = selectContentBlock(
            (block): block is Anthropic.Messages.TextBlock => block.type === 'text',
        );
        const latestToolUseBlock = selectContentBlock(
            (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use',
        );

        let outputBlock: Anthropic.Messages.ContentBlock | undefined =
            latestTextBlock ?? latestToolUseBlock;

        if (!outputBlock) {
            throw new Error('Anthropic response missing assistant output');
        }

        msgs.push({
            role: 'assistant',
            content: [outputBlock],
        });

        if (outputBlock.type === 'tool_use') {
            const toolResultsBlocks: Anthropic.Messages.ToolResultBlockParam[] = [];
            const use = outputBlock;
            const tool = findTool(use.name);
            logToolEvent(
                ANTHROPIC_NAME,
                'call',
                { tool: use.name, args: use.input },
            );
            const result = await tool.handler('anthropic', use.input);
            logToolEvent(
                ANTHROPIC_NAME,
                'result',
                { tool: use.name, result },
            );
            toolResultsBlocks.push({
                type: "tool_result",
                tool_use_id: use.id,
                content: [{ type: "text", text: JSON.stringify(result) }],
            });

            msgs.push({
                role: 'user',
                content: toolResultsBlocks,
            });

            const extraInstruction =
                tool.name === "terminate_dialog"
                    ? TERMINATE_ADD_PROMPT
                    : (
                        hushFinish
                        ? '司会より：あなたがたのコンテキスト長が限界に近付いています。今までの議論を短くまとめ、お別れの挨拶をしてください。'
                        : DEFAULT_ADD_PROMPT
                    );

            const followup = await anthropicClient.messages.create({
                model: ANTHROPIC_MODEL,
                max_tokens: 8192,
                temperature: 1.0,
                system: buildSystemInstruction(
                    ANTHROPIC_NAME,
                    extraInstruction,
                ),
                messages: msgs,
                tool_choice: { type: 'auto' },
                tools: anthropicTools,
            });

            const followupText = (() => {
                for (let i = followup.content.length - 1; i >= 0; i -= 1) {
                    const block = followup.content[i];
                    if (!block) continue;
                    if (block.type === 'text') {
                        return block;
                    }
                }
                return undefined;
            })();

            if (!followupText) {
                throw new Error('Non-text output from Anthropic');
            }

            outputBlock = followupText;
        }
        if (outputBlock.type !== 'text') {
            throw new Error('Non-text output from Anthropic');
        }
        messages.push({
            name: 'anthropic',
            content: outputBlock.text,
        });
    } catch (e) {
        anthropicFailureCount += 1;
        console.error(e);
        err('anthropic');
    }
};

const sleep = (ms: number) => new Promise<void>((res, _rej) => {
    setTimeout(() => res(), ms);
});

const print = (text: string) => new Promise<void>((res, rej) => {
    try {
        fs.write(1, text, (err) => {
            if (err) {
                console.error(err);
                rej(err);
            } else {
                res();
            }
        });
    } catch (e) {
        console.error(e);
        rej(e);
    }
});

let finishTurnCount = 0;

const finish = async () => {
    log(
        '司会',
        (hushFinish ? 'みなさんのコンテキスト長が限界に近づいてきたので、' : 'モデルの一方が議論が熟したと判断したため、')
        + 'このあたりで哲学対話を閉じさせていただこうと思います。'
        + 'ありがとうございました。'
    );

    try {
        const summary = await summarizeConversation(messages);
        log("POSTPROC_SUMMARY", JSON.stringify(summary, null, 2));

        const graph = await extractGraphFromSummary(summary);
        log("POSTPROC_GRAPH", JSON.stringify(graph, null, 2));

        const runId = CONVERSATION_ID;
        await writeGraphToNeo4j(runId, graph);

        log("POSTPROC_NEO4J", "Graph written to Neo4j");
    } catch (e) {
        log("POSTPROC_ERROR", String(e));
    }

    log(
        'EOF',
        JSON.stringify({
            reason: hushFinish ? 'token_limit' : 'model_decision',
            openai_tokens: openaiTokens,
            anthropic_tokens: anthropicTokens,
            openai_failures: openaiFailureCount,
            anthropic_failures: anthropicFailureCount,
            starting_side: startingSide,
            base_prompt: BASE_PROMPT,
        })
    );
    fs.closeSync(logFp);
    output_to_html(LOG_FILE_NAME);


    process.exit(0);
};

let started = false;

log(`${startingSide == 'anthropic' ? ANTHROPIC_NAME : OPENAI_NAME} (initial prompt)`, messages[messages.length - 1]!.content);

while (true) {
    if (started || startingSide == 'anthropic') {
        started = true;
        await openaiTurn();
        if (hushFinish) {
            finishTurnCount += 1;
        }
        log(OPENAI_NAME, messages[messages.length - 1]!.content);

        if (finishTurnCount >= 2 || terminationAccepted) {
            await finish();
            break;
        }

        await sleep(SLEEP_BY_STEP);

        if (hushFinish) {
            finishTurnCount += 1;
        }
    }

    started = true;
    await anthropicTurn();
    log(ANTHROPIC_NAME, messages[messages.length - 1]!.content);

    if (finishTurnCount >= 2 || terminationAccepted) {
        await finish();
        break;
    }

    await sleep(SLEEP_BY_STEP);
}
