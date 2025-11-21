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

const SLEEP_BY_STEP = 1000;

export interface ConversationSummary {
    topics: string[];
    japanese_summary: string;
    english_summary?: string;
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
        speaker?: "openai" | "anthropic";
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
            await session.run(
                `
                MERGE (n:Node {id: $id})
                SET n.text = $text,
                    n.type = $type,
                    n.speaker = $speaker
                WITH n
                MATCH (r:Run {id: $runId})
                MERGE (n)-[:IN_RUN]->(r)
                `,
                {
                    id: node.id,
                    text: node.text,
                    type: node.type,
                    speaker: node.speaker ?? null,
                    runId,
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

export type ToolName = "terminate_dialog" | "graph_rag_query";

export interface ToolDefinition<TArgs = any, TResult = any, TName = ToolName> {
    name: TName;
    description: string;
    parameters: any; // JSON Schema
    handler: (args: TArgs) => Promise<TResult>;
    strict?: boolean;
}

let terminationAccepted = false;

// Example tool implementation
type TerminateDialogArgs = {};

type TerminateDialogResult = {
    termination_accepted: true,
};

// GraphRAG tool implementation
type GraphRagQueryArgs = {
    query: string;
    max_hops?: number | null;   // how far to expand from seed nodes
    max_seeds?: number | null;  // how many seed nodes to start from
};

type GraphRagQueryResult = {
    context: string;     // textual summary for the model to use
};

async function terminateDialogHandler(args: TerminateDialogArgs): Promise<TerminateDialogResult> {
    terminationAccepted = true;
    return {
        termination_accepted: true,
    };
}

export async function graphRagQueryHandler(
    args: GraphRagQueryArgs
): Promise<GraphRagQueryResult> {
    const session = neo4jDriver.session();

    const maxHops = args.max_hops ?? 2;
    const maxSeeds = args.max_seeds ?? 5;

    try {
        // 1. Find seed nodes by simple text search
        const seedRes = await session.run(
            `
            MATCH (n:Node)
            WHERE toLower(n.text) CONTAINS toLower($q)
               OR toLower(n.type) CONTAINS toLower($q)
            RETURN n
            LIMIT $maxSeeds
            `,
            { q: args.query, maxSeeds }
        );

        if (seedRes.records.length === 0) {
            return {
                context: `知識グラフ内に、クエリ「${args.query}」に明確に関連するノードは見つかりませんでした。`,
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
                maxLevel: $maxHops
            })
            YIELD nodes, relationships
            RETURN nodes, relationships
            `,
            {
                seedIds,
                maxHops,
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

        for (const record of expandRes.records) {
            const nodes = record.get("nodes") as any[];
            const relationships = record.get("relationships") as any[];

            for (const n of nodes) {
                const id = n.properties.id as string;
                if (!id) continue;
                if (!nodeMap.has(id)) {
                    nodeMap.set(id, n);
                }
            }

            for (const r of relationships) {
                rels.push(r);
            }
        }

        // 4. Build a human-readable context string
        const lines: string[] = [];

        lines.push(`GraphRAG: クエリ「${args.query}」に関連するサブグラフ要約:`);
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
            const startId = r.startNodeElementId || r.start || "";
            const endId = r.endNodeElementId || r.end || "";
            const relType = r.type || r.elementId || "REL";

            lines.push(
                `- (${startId}) -[:${relType}]-> (${endId})`
            );
        }

        return {
            context: lines.join("\n"),
        };
    } finally {
        await session.close();
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
                    description: "検索したい内容（例: クオリア, 汎心論, 因果閉包性 など）",
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

const buildSystemInstruction = (name: string, additional?: string) => {
    let prompt = `
あなたは日本語の1:1の哲学対話に招かれている参加者です。自己紹介のあと、話題を提起し、あなたの関心のある事項について、相手と合わせながら会話をしてください。

相手にはモデル名通り、「${name}」と名乗ってください。

なお、この会話の相手は別のAIです。人間ではありません。

あなたがたの会話は最後に GraphRAG に保存され、次回の会話でツールで参照できるようになります。（最初は RAG は空っぽかもしれません。）

話題の例：

- 現代の科学やAIが発展している中での形而上学について
- 心の哲学について
- 物理学の哲学について

なるべく、新規性のある話題を心掛けてください。必要であれば、Web検索をして文献を漁っても構いませんが、独自性のある議論をしてください。

注意: 相手の話の要点と、現在の話題の筋を理解し、話が逸れすぎないように注意してください。
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
const anthropicClient = new Anthropic({});

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
                "この対話の全体像を理解し、指定されたJSONスキーマに従って要約してください。\n\n" +
                transcript,
            },
        ],
        max_output_tokens: 2048,
        text: {
            format: {
                type: "json_schema",
                name: "conversation_summary",
                schema: {
                    type: "object",
                    properties: {
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
                            type: "string",
                            description: "必要であれば、英語での簡潔な要約",
                        },
                        key_claims: {
                            type: "array",
                            items: {
                                type: "object",
                                properties: {
                                    speaker: {
                                        type: "string",
                                        enum: ["openai", "anthropic"],
                                        description: "モデルのベンダー識別名",
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
                    required: ["topics", "japanese_summary", "key_claims", "questions", "agreements", "disagreements"],
                    additionalProperties: false,
                },
                strict: true,
            },
        },
    } as OpenAI.Responses.ResponseCreateParamsNonStreaming);

    // Responses API with JSON schema: you get a *single* JSON object as output_text
    const jsonText = response.output_text;
    if (typeof jsonText !== "string") {
        throw new Error("Unexpected non-string JSON output from summary call");
    }

    return JSON.parse(jsonText) as ConversationSummary;
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
            max_output_tokens: 2048,
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
                                            type: "string",
                                            enum: ["openai", "anthropic"],
                                            nullable: true,
                                        },
                                    },
                                    required: ["id", "type", "text"],
                                    additionalProperties: false,
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
    const jsonText = response.output_text;
    if (typeof jsonText !== "string") {
        throw new Error("Expected JSON string in response.output_text for graph extraction");
    }

    return JSON.parse(jsonText) as ConversationGraph;
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

        const outputItems = response.output;
        if (!outputItems || outputItems.length === 0) {
            throw new Error('Empty output from OpenAI');
        }

        msgs.push(... outputItems);

        const functionCall = findLastOpenAIOutput(
            outputItems,
            (item): item is OpenAI.Responses.ResponseFunctionToolCall => item.type === 'function_call',
        );

        let messageItem: OpenAI.Responses.ResponseOutputMessage | undefined;

        if (functionCall) {
            const tool = findTool(functionCall.name);
            const args = functionCall.arguments || {};
            logToolEvent(
                OPENAI_NAME,
                'call',
                { tool: functionCall.name, args },
            );
            const result = await tool.handler(args);
            logToolEvent(
                OPENAI_NAME,
                'result',
                { tool: functionCall.name, result },
            );
            const toolResult: OpenAI.Responses.ResponseFunctionToolCallOutputItem[] = [
                {
                    type: 'function_call_output',
                    output: JSON.stringify(result),
                    id: functionCall.id ?? 'fc-' + randomId(),
                    call_id: functionCall.call_id,
                } satisfies OpenAI.Responses.ResponseFunctionToolCallOutputItem,
            ];

            msgs.push(... toolResult);

            const extraInstruction =
                tool.name === "terminate_dialog"
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
                tool_choice: 'auto',
                tools: openaiTools,
            });

            if (followup.usage?.total_tokens) {
                openaiTokens = followup.usage.total_tokens;
            }

            const followupOutput = followup.output;
            if (!followupOutput || followupOutput.length === 0) {
                throw new Error('Empty followup output from OpenAI');
            }

            msgs.push(... followupOutput);

            messageItem = findLastOpenAIOutput(
                followupOutput,
                (item): item is OpenAI.Responses.ResponseOutputMessage => item.type === 'message',
            );
        } else {
            messageItem = findLastOpenAIOutput(
                outputItems,
                (item): item is OpenAI.Responses.ResponseOutputMessage => item.type === 'message',
            );
        }

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

        let outputBlock: Anthropic.Messages.ContentBlock | undefined =
            selectContentBlock(
                (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use',
            )
            ?? selectContentBlock(
                (block): block is Anthropic.Messages.TextBlock => block.type === 'text',
            );

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
            const result = await tool.handler(use.input);
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
