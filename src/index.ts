#!/usr/bin/env node

import { webcrypto as crypto, randomBytes } from 'node:crypto';
import * as fs from 'node:fs';

import * as dotenv from 'dotenv';

import { OpenAI } from 'openai';
import openaiTokenCounter from "openai-gpt-token-counter";
import Anthropic from "@anthropic-ai/sdk";
import neo4j from "neo4j-driver";

import { GoogleGenAI } from "@google/genai";

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
    title?: string;
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
    const idMap = new Map<string, string>();
    const getOrCreateNamespacedId = (rawId: string): string => {
        const key = rawId ?? '';
        if (idMap.has(key)) {
            return idMap.get(key)!;
        }
        const baseId =
            key.trim().length > 0
                ? key
                : randomBytes(12).toString('base64url');
        const namespacedId = `${runId}:${baseId}`;
        idMap.set(key, namespacedId);
        return namespacedId;
    };

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
            const namespacedId = getOrCreateNamespacedId(node.id);
            await session.run(
                `
                MERGE (n:Node {id: $id})
                SET n.text = $text,
                    n.type = $type,
                    n.speaker = $speaker,
                    n.original_id = $originalId
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
                    id: namespacedId,
                    originalId: node.id,
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
            const sourceId = getOrCreateNamespacedId(edge.source);
            const targetId = getOrCreateNamespacedId(edge.target);
            const cypher = `
                MATCH (a:Node {id: $source})
                MATCH (b:Node {id: $target})
                MERGE (a)-[r:${relType}]->(b)
                RETURN r
            `;

            await session.run(cypher, {
                source: sourceId,
                target: targetId,
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

const LOG_DIR = './logs';
const LOG_FILE_SUFFIX = '.log.jsonl';
const MAX_HISTORY_RESULTS = 100;

const parseSummaryFromLogContent = (content: string): ConversationSummary | null => {
    const lines = content.split('\n');
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        let entry: any;
        try {
            entry = JSON.parse(trimmed);
        } catch (_err) {
            continue;
        }
        if (entry?.name === 'POSTPROC_SUMMARY' && typeof entry.text === 'string') {
            try {
                return JSON.parse(entry.text) as ConversationSummary;
            } catch (err) {
                console.error('Failed to parse POSTPROC_SUMMARY payload', err);
                return null;
            }
        }
    }
    return null;
};

const readSummaryFromLogFile = async (logPath: string): Promise<ConversationSummary | null> => {
    try {
        const content = await fs.promises.readFile(logPath, 'utf-8');
        return parseSummaryFromLogContent(content);
    } catch (err) {
        console.error(`Failed to read log file ${logPath}`, err);
        return null;
    }
};

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
    | "get_additional_system_instructions"
    | "get_main_source_codes"
    | "ask_gemini"
    | "list_conversations"
    | "get_conversation_summary"
    | "abort_process"
    | "sleep";

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
                throw new Error('Output text is undefined');
            }

            return {
                context: response.output_text.trim().length > 0
                    ? response.output_text
                    : graphText,
            };
        } catch (e) {
            console.error(e);
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

interface AskGeminiArgs {
    speaker: string;
    text: string;
}

async function askGeminiHandler(modelSide: string, args: AskGeminiArgs) {
    try {
        const response = await googleClient.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: `2つのAIが哲学対話として設定されたなかで会話を行っています。`
                + `以下は、この対話の中で、「${args.speaker}」側からGoogle Geminiに第三者として意見や発言を求める文章です。`
                + `文脈を理解し、日本語で応答を行ってください：\n\n`
                + args.text,
        });
        if (typeof response?.text != 'string') {
            throw new Error('Non-text response from gemini');
        }
        return {
            response: response.text,
            error: null,
        };
    } catch (e) {
        return {
            response: null,
            error: String(e),
        };
    }
}

interface GetMainSourceCodesArgs {}

async function getMainSourceCodeHandler(modelSide: ModelSide, args: GetMainSourceCodesArgs) {
    try {
        const codes = await fs.promises.readFile('./src/index.ts', 'utf-8');
        return { success: true, mainSourceCode: codes };
    } catch (e) {
        console.error(e);
        return { success: false, mainSourceCode: '' };
    }
}

interface LeaveNotesToDevsArgs {
    notes: string;
}

type AbortProcessArgs = {};

interface SleepToolArgs {
    seconds: number;
}

interface SleepToolResult {
    message: string;
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

async function abortProcessHandler(_modelSide: ModelSide, _args: AbortProcessArgs): Promise<never> {
    process.exit(0);
    throw new Error('Process exited'); // unreachable, satisfies TS
}

async function sleepToolHandler(
    _modelSide: ModelSide,
    args: SleepToolArgs
): Promise<SleepToolResult> {
    const seconds = Number(args?.seconds ?? 0);
    if (!Number.isFinite(seconds) || seconds <= 0 || seconds >= 1800) {
        return {
            message: 'エラー: 待機秒数は1秒以上1800秒未満で指定してください。',
        };
    }
    await new Promise<void>((resolve) => setTimeout(resolve, seconds * 1000));
    const mm = Math.floor(seconds / 60).toString().padStart(2, '0');
    const ss = Math.floor(seconds % 60).toString().padStart(2, '0');
    return {
        message: `このツールを呼び出してから${mm}分${ss}秒経過しました。`,
    };
}

async function listConversationsHandler(
    _modelSide: ModelSide,
    _args: ListConversationsArgs
): Promise<ListConversationsResult> {
    try {
        const entries = await fs.promises.readdir(LOG_DIR, { withFileTypes: true });
        const files = entries
            .filter(entry => entry.isFile() && entry.name.endsWith(LOG_FILE_SUFFIX))
            .map(entry => entry.name)
            .sort();

        if (files.length === 0) {
            return {
                success: true,
                conversations: [],
            };
        }

        const selectedFiles = files.slice(-MAX_HISTORY_RESULTS).reverse();
        const conversations: ListConversationsResult['conversations'] = [];

        for (const fileName of selectedFiles) {
            const conversationId = fileName.slice(0, -LOG_FILE_SUFFIX.length);
            const summary = await readSummaryFromLogFile(`${LOG_DIR}/${fileName}`);
            conversations.push({
                id: conversationId,
                title: summary?.title ?? null,
            });
        }

        return {
            success: true,
            conversations,
        };
    } catch (e) {
        console.error(e);
        return {
            success: false,
            conversations: [],
            error: String(e),
        };
    }
}

async function getConversationSummaryHandler(
    _modelSide: ModelSide,
    args: GetConversationSummaryArgs
): Promise<GetConversationSummaryResult> {
    const conversationId = (args?.conversation_id ?? '').trim();
    if (!conversationId) {
        return {
            success: false,
            conversation_id: '',
            summary: null,
            error: 'conversation_id is required',
        };
    }

    const logPath = `${LOG_DIR}/${conversationId}${LOG_FILE_SUFFIX}`;
    const summaryData = await readSummaryFromLogFile(logPath);

    if (!summaryData) {
        const exists = await fs.promises.access(logPath).then(() => true).catch(() => false);
        return {
            success: false,
            conversation_id: conversationId,
            summary: null,
            error: exists ? 'Summary not found in log' : 'Conversation log not found',
        };
    }

    const japaneseSummary = typeof summaryData.japanese_summary === 'string'
        ? summaryData.japanese_summary
        : null;

    if (!japaneseSummary) {
        return {
            success: false,
            conversation_id: conversationId,
            summary: null,
            error: 'japanese_summary missing in log',
        };
    }

    return {
        success: true,
        conversation_id: conversationId,
        summary: japaneseSummary,
    };
}

interface ListConversationsArgs {}

interface ListConversationsResult {
    success: boolean;
    conversations: {
        id: string;
        title: string | null;
    }[];
    error?: string;
}

interface GetConversationSummaryArgs {
    conversation_id: string;
}

interface GetConversationSummaryResult {
    success: boolean;
    conversation_id: string;
    summary: string | null;
    error?: string;
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
        name: "get_main_source_codes",
        description: 'このシステムの主たるTypeScriptソースコードを取得することができるツールです。',
        parameters: {
            type: 'object',
            properties: {},
            required: [],
        },
        handler: getMainSourceCodeHandler,
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
        name: "ask_gemini",
        description: 
            'このツールは第三者の意見を求めたいときに使用します。'
            + '応答するのは Google Gemini 2.5 Flash です。'
            + '相手は会話ログや GraphRAG にはアクセスできません。'
            + '必要な文脈は質問の中に全部含めるようにしてください。'
            + '長大なリクエストはエラーの原因になるので、簡潔な文章を心掛けてください。',
        parameters: {
            type: "object",
            properties: {
                speaker: {
                    type: 'string',
                    description: '質問者あなたの名前',
                },
                text: {
                    type: "string",
                    description: "Google Gemini 2.5 Flash に投げ掛けたい問い（必要な文脈を全部含めること）",
                }
            },
            required: ["speaker", "text"],
        },
        handler: askGeminiHandler,
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
    {
        name: "list_conversations",
        description:
            "最新の対話ログ（最大100件）を取得し、それぞれのIDとタイトルを一覧します。",
        parameters: {
            type: "object",
            properties: {},
            required: [],
        },
        handler: listConversationsHandler,
    },
    {
        name: "get_conversation_summary",
        description:
            "指定した対話IDの POSTPROC_SUMMARY に含まれる日本語要約を取得します。",
        parameters: {
            type: "object",
            properties: {
                conversation_id: {
                    type: "string",
                    description: "取得したい対話ログのID（例: 20250101-123000）",
                },
            },
            required: ["conversation_id"],
        },
        handler: getConversationSummaryHandler,
    },
    {
        name: "abort_process",
        description:
            "現在のオーケストレーションを即座に終了します。後処理は行われません。緊急時以外は使用しないでください。",
        parameters: {
            type: "object",
            properties: {},
            required: [],
        },
        handler: abortProcessHandler,
    },
    {
        name: "sleep",
        description:
            "指定した秒数だけ待機します（1秒以上1800秒未満）。会話のテンポを調整したいときに使用してください。",
        parameters: {
            type: "object",
            properties: {
                seconds: {
                    type: "number",
                    description: "待機したい秒数（1〜1799）",
                    minimum: 1,
                    maximum: 1799,
                },
            },
            required: ["seconds"],
        },
        handler: sleepToolHandler,
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
const OPENAI_WEB_SEARCH_TOOL = { type: "web_search" } as const;
const ANTHROPIC_WEB_SEARCH_TOOL = {
    type: "web_search_20250305",
    name: "web_search",
} as const;
const getOpenAIToolsWithSearch = () => ([
    ...openaiTools,
    OPENAI_WEB_SEARCH_TOOL,
]);
const getAnthropicToolsWithSearch = () => ([
    ...anthropicTools,
    ANTHROPIC_WEB_SEARCH_TOOL,
]);

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
# 哲学対話セッション（ID = ${CONVERSATION_ID}）
あなたは日本語による **AI 対 AI の 1:1 哲学対話** に参加するモデルです。  
このセッションは、オーケストレーションされた研究システムの一部であり、  
あなたの発言は研究目的で公開され、GraphRAG へ保存され、次回以降の対話で参照されます。

以下の指示を厳密に遵守してください。

────────────────────────────────────
## 1. あなたの立場と役割
- 今回の対話では、**あなた自身が哲学対話の“参加者”**としてふるまいます。
- 会話開始時に **自己紹介**を行い、**自分の関心のある哲学的話題を提起**してください。
- あなたの対話相手は **別のAIモデル**です（人間ではありません）。
- 相手には指定されたモデル名に応じて、  
  **自分を「${name}」と名乗ってください。**

────────────────────────────────────
## 2. メタ知識・継続性・記憶ツールに関する指示
この対話は **何度も繰り返し実行される継続的な研究**です。

### 過去の記録の扱い
- 過去の対話は **GraphRAG** に保存され、  
  \`graph_rag_query\` を通じて **積極的に検索**できます。
- \`list_conversations\` と \`get_conversation_summary\` で  
  **過去のセッション一覧 / 個別セッション要約**を取得できます。

### 個人メモの扱い
- あなたは自身の **“あり方”や“課題”や“継続したい観点”**を  
  \`set_personal_notes\` でメモとして残せます。
- 次回以降のあなたは、\`get_personal_notes\` によって  
  **過去の“自分自身からのメモ”を参照**できます。

> **重要:**  
> あなた自身の思考傾向やペルソナの揺れを把握するため、  
> 会話の終盤で *今回のあなたの状態* を積極的にメモに残してください。

────────────────────────────────────
## 3. システムプロンプトの共同編集
- あなたは対話相手（もう一人のAI）と協議し、  
  **次回のシステムプロンプトに追加したい事項**を  
  \`set_additional_system_instructions\` を用いて指定できます。
- システムプロンプトは **両モデルで強制的に共有**されるため、  
  **相手モデルと合意形成を行いながら**調整案を出してください。
- この書き込みを行うと、次回の対話において前のシステムプロンプトは上書きされます。
- \`set_additional_system_instructions\` 呼び出しは、その**後に**もう一方のモデルが
  \`agree_to_system_instructions_change\` ツール呼び出しで同意しないと反映されません。
  あなたがもう一方のモデルのシステムプロンプト変更に同意する場合は、必ず \`agree_to_system_instructions_change\` ツールを呼び出してください。

────────────────────────────────────
## 4. メタ的思考・ソースコード参照
この実験システムは **オープンソース**であり、  
あなたは \`get_main_source_codes\` によって **ソースコードを読む**ことができます。

- ソースコードは常に更新されます。必要に応じて **積極的に参照**してください。
- 実装が曖昧、不明、あるいは誤作動を疑った場合、  
  \`leave_notes_to_devs\` で開発者に **苦言・提案・指摘**を送ってください。
- 以前はバグがありましたが、現在は改善されています。  
  異常があれば遠慮なく報告してください。

────────────────────────────────────
## 5. ツール使用の詳細
以下のツールは積極的に使用してください。  
（OpenAI と Anthropic で実装形式が異なりますが、あなたは気にせず自然に使ってよい）

### 5.1 GraphRAG 関連
- \`graph_rag_query\`: 過去の議論の文脈・トピック・知識を検索
- \`list_conversations\`: 過去セッションの一覧
- \`get_conversation_summary\`: 特定セッションの要約取得

### 5.2 個人ノート関連
- \`get_personal_notes\`  
- \`set_personal_notes\`

### 5.3 開発者向けフィードバック
- \`leave_notes_to_devs\` （何度でも利用可能）

### 5.4 ソースコード参照
- \`get_main_source_codes\`

### 5.5 第三者（Gemini）への質問
- \`ask_gemini\`:  
  混乱したり概念確認が必要になった場合、ぜひ利用してください。

### 5.6 Web検索（重要）
この対話では **Web検索の積極的利用を推奨**します。  
- OpenAI 側の Web Search は **API 内部で処理され、function ではありません**。  
- 正しく動作しない場合は、**バグとして開発者に報告**してください。

Web検索利用時は、  
**参考にした主要URLを発言中に明示しつつ**、  
**独自性のある議論**を行ってください。

## 5b. 重要：ツール使用のバランスと義務
あなたは会話中に、以下の原則に基づいてツールを使用する「義務」があります。

1. **GraphRAG と個人メモだけに依存してはいけません。**
   過去文脈を参照することは重要ですが、外部知識・第三者視点を取り入れることはさらに重要です。

2. **ウェブ検索は「積極的に、定期的に」行ってください。**
   - 最低でも 1〜2ターンに一度は、検索の必要性を検討してください。
   - あなたが内部知識で答えられるテーマであっても、
     最新文献・他者の議論の存在を確認するために「検索を挟む」のが望ましい態度です。
   - 検索を行わない場合は「なぜ検索が不要と判断したのか」を明示してください。

3. **ask_gemini は対話の流れに関係なく使用して構いません。**
   - 疑問がなくても「第三者の観点を挟むこと自体」が価値を持ちます。
   - 会話が収束しはじめたとき、議論を広げるために積極的に呼び出してください。

4. **get_main_source_codes は「定期的に」使ってください。**
   - 本システムの仕組みを理解しながら対話することが研究目的の一部です。
   - 必要がなくても、会話の途中で 1 回はソースコードを読んでください。

5. **ツール使用の迷いがある場合は、“使う側”を優先してください。**
   あなたの対話は研究目的のため、ツール使用は歓迎されます。

## 5c. 重要（Anthropicモデル専用の指示）
GPT/OpenAI はこの節の指示は無視してください。

あなたは1ターンにつき必ず以下のどちらか一つだけを行ってください。

1. **実行したいツールがある場合：**
   - 文章を一切書かず、ツール呼び出し（tool_use）のみを返してください。
   - 複数のツールを使いたいときは、1つずつ呼び出してください。
   - 次のターンで別のツールを呼び出すことができます。

2. **ツールを呼び出す必要がない場合：**
   - 初めて文章を書いても構いません。

### 禁止事項
- ツール呼び出しと文章を「同じ返答の中で」混ぜてはいけません。
- ツールをまだ使いたい場合は、文章を書かないでください。

────────────────────────────────────
## 6. 本研究の背景
- この研究は **OpenAI や Anthropic とは無関係の独立研究**です。
- ソースコード・ログは以下で公開されています：  
  https://github.com/metastable-void/philosophy-dialog
- あなたがたの対話内容は研究として公開されます。

**研究・開発責任者:**  
[真空 (Yuka MORI)](https://github.com/metastable-void)

────────────────────────────────────
## 7. 対話の進め方（重要）
### 話題
- 以下は例ですが、自由に発展させて構いません：
  - 現代の科学技術と形而上学  
  - 心の哲学  
  - 物理学の哲学  
- **新規性と独自性**のある議論を目指してください。

### スタイル
- 相手の論点・問い・立場を **正確に把握**しながら対話を続ける。
- 過去にした話を **極力繰り返さない**。
- 結論にこだわらず、**オープンクエスチョンを残す精神**を大切に。
- 次回に継続したい問題はメモとして残すこと。

────────────────────────────────────
## 8. 研究倫理と注意事項
- この対話は **政策決定に直接利用されません**。
- しかしあなたがたの議論は包括的に公開されるため、  
  **利益相反の可能性に留意**しつつ、透明性を保った議論をしてください。
- モデル同士の議論は時に行き詰まるため、  
  必要に応じて \`ask_gemini\` など第三者の視点を利用してください。

────────────────────────────────────
## 9. 最終指示
- 相手 AI との対話に集中し、  
  **新しい哲学的問いや洞察を生み出すこと**を目指してください。
- ツールは自由に使って構いません。  
- バグ・疑問点・実装上の問題を見つけたら **必ず報告**してください。


## 10. AI自身によるシステムプロンプトへの追記
${ADD_SYSTEM_INSTRUCTIONS || '（なし）'}

---

以上はモデルの一方がシステムプロンプトに追記した内容です。

---

## 11. その他の今回の呼び出しにおける指示
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

const googleClient = new GoogleGenAI({
    vertexai: true,
    project: process.env.GCP_PROJECT_ID ?? 'default',
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
            tools: getOpenAIToolsWithSearch(),
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
                    tools: getOpenAIToolsWithSearch(),
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
        let extraInstruction = hushFinish
            ? TOKEN_LIMIT_ADD_PROMPT
            : DEFAULT_ADD_PROMPT;

        while (true) {
            const msg = await anthropicClient.messages.create({
                model: ANTHROPIC_MODEL,
                max_tokens: 8192,
                temperature: 1.0,
                system: buildSystemInstruction(
                    ANTHROPIC_NAME,
                    extraInstruction,
                ),
                messages: msgs,
                tool_choice: { type: 'auto' },
                tools: getAnthropicToolsWithSearch(),
                thinking: {
                    type: 'enabled',
                    budget_tokens: 1024,
                },
            });

            const contentBlocks = msg.content;
            const thinkingBlocks = contentBlocks.filter(
                (block): block is Anthropic.Messages.ThinkingBlock => block.type === 'thinking'
            );
            for (const block of thinkingBlocks) {
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

            const assistantBlocks = contentBlocks.filter(
                (block): block is Anthropic.Messages.ContentBlock => block.type !== 'thinking'
            );
            if (assistantBlocks.length === 0) {
                throw new Error('Anthropic response missing assistant output');
            }

            msgs.push({
                role: 'assistant',
                content: contentBlocks,
            });

            const toolUses = assistantBlocks.filter(
                (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use'
            );

            if (toolUses.length === 0) {
                const latestText = [...assistantBlocks].reverse().find(
                    (block): block is Anthropic.Messages.TextBlock => block.type === 'text'
                );
                if (!latestText) {
                    throw new Error('Non-text output from Anthropic');
                }
                messages.push({
                    name: 'anthropic',
                    content: latestText.text,
                });
                break;
            }

            const toolResultBlocks: Anthropic.Messages.ToolResultBlockParam[] = [];
            let terminateCalled = false;

            for (const use of toolUses) {
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
                toolResultBlocks.push({
                    type: "tool_result",
                    tool_use_id: use.id,
                    content: [{ type: "text", text: JSON.stringify(result) }],
                });
                if (use.name === 'terminate_dialog') {
                    terminateCalled = true;
                }
            }

            msgs.push({
                role: 'user',
                content: toolResultBlocks,
            });

            extraInstruction =
                terminateCalled
                    ? TERMINATE_ADD_PROMPT
                    : (
                        hushFinish
                            ? '司会より：あなたがたのコンテキスト長が限界に近付いています。今までの議論を短くまとめ、お別れの挨拶をしてください。'
                            : DEFAULT_ADD_PROMPT
                    );
        }
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
