#!/usr/bin/env node

import { webcrypto as crypto, randomBytes } from 'node:crypto';
import * as fs from 'node:fs';
import * as url from 'node:url';
import * as path from 'node:path';

import * as dotenv from 'dotenv';

import { OpenAI } from 'openai';
import openaiTokenCounter from "openai-gpt-token-counter";
import Anthropic from "@anthropic-ai/sdk";
import neo4j from "neo4j-driver";

import { GoogleGenAI } from "@google/genai";

import { output_to_html } from './html.js';

const FILENAME = fs.realpathSync(url.fileURLToPath(import.meta.url));
const ARGV1 = fs.realpathSync(path.resolve(process.argv[1]!));

/// false if we are imported from other scripts
export const IS_MAIN = ARGV1 === FILENAME;

/// INITIALIZE ENV
if (IS_MAIN) dotenv.config();

/// CONST DEFINITIONS
const GPT_5_1_MAX = 400000;
const CLAUDE_HAIKU_4_5_MAX = 200000;
const STRUCTURED_OUTPUT_MAX_TOKENS = 16384;
const CONCEPT_LINK_REL = 'NORMALIZED_AS';

const SLEEP_BY_STEP = 1000;
const DEFAULT_LOG_DIR = './logs';
const DEFAULT_DOCS_DIR = './docs';
const LOG_FILE_SUFFIX = '.log.jsonl';
const MAX_HISTORY_RESULTS = 100;
const DEFAULT_DATA_DIR = './data';
const TOOL_STATS_SUBDIR = 'tool-stats';
const PENDING_SYSTEM_INSTRUCTIONS_FILENAME = 'pending-system-instructions.json';

const DEFAULT_OPENAI_MODEL = 'gpt-5.1';
const DEFAULT_ANTHROPIC_MODEL = 'claude-haiku-4-5';

const DEFAULT_OPENAI_NAME = 'GPT 5.1';
const DEFAULT_ANTHROPIC_NAME = 'Claude Haiku 4.5';

/// TYPES
export interface PhilosophyDialogArgs {
    logDir: string;
    dataDir: string;
    docsDir: string;
    openaiModel: string;
    anthropicModel: string;
    openaiName: string;
    anthropicName: string;
}

type ModelSide = 'openai' | 'anthropic';

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

interface Data {
    personalNotes: string;
    additionalSystemInstructions: string;
}

interface PendingSystemInstructions {
    instructions: string;
    requestedBy: ModelSide;
    createdAt: string;
}

type ToolUsageStats = Record<string, Record<string, number>>;

export type ToolName =
    "terminate_dialog"
    | "graph_rag_query"
    | "graph_rag_focus_node"
    | "get_personal_notes"
    | "set_personal_notes"
    | "leave_notes_to_devs"
    | "set_additional_system_instructions"
    | "get_additional_system_instructions"
    | "agree_to_system_instructions_change"
    | "get_main_source_codes"
    | "ask_gemini"
    | "list_conversations"
    | "get_conversation_summary"
    | "compare_conversation_themes"
    | "get_tool_usage_stats"
    | "abort_process"
    | "sleep";

export interface ToolDefinition<TArgs = any, TResult = any, TName = ToolName> {
    name: TName;
    description: string;
    parameters: any; // JSON Schema
    handler: (modelSide: ModelSide, args: TArgs) => Promise<TResult>;
    strict?: boolean;
}

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

type GraphRagFocusNodeArgs = {
    node_id: string;
    max_hops?: number | null;
};

type GraphRagFocusNodeResult = {
    context: string;
};

interface GetAdditionalSystemInstructionsArgs {}

interface SetAdditionalSystemInstructionsArgs {
    systemInstructions: string;
}

interface AgreeSystemInstructionsArgs {}

interface AskGeminiArgs {
    speaker: string;
    text: string;
}

interface GetMainSourceCodesArgs {}

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

interface CompareConversationThemesArgs {
    conversation_ids: string[];
}

interface CompareConversationThemesResult {
    success: boolean;
    comparisons?: {
        conversation_id: string;
        title: string | null;
        topics: string[];
        japanese_summary: string;
    }[];
    analysis?: {
        common_themes: string[];
        divergences: string[];
        emerging_questions: string[];
    };
    errors?: string[];
    error?: string;
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

interface GetToolUsageStatsArgs {
    conversation_id: string;
}

interface GetToolUsageStatsResult {
    success: boolean;
    conversation_id: string;
    stats: ToolUsageStats | null;
    error?: string;
}


/// HELPERS
const randomBoolean = (): boolean => {
    const b = new Uint8Array(1);
    crypto.getRandomValues(b);
    return (b[0]! & 1) == 1;
};

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

const aggregateToolStats = (entries: any[]): ToolUsageStats => {
    const stats: ToolUsageStats = {};
    for (const entry of entries) {
        if (!entry || typeof entry.name !== 'string') continue;
        if (!entry.name.endsWith(' (tool call)')) continue;
        let payload: any;
        try {
            payload = JSON.parse(entry.text);
        } catch (_e) {
            continue;
        }
        const toolName = payload?.tool;
        if (!toolName) continue;
        const actor = entry.name.replace(/ \(tool call\)$/, '');
        stats[actor] = stats[actor] ?? {};
        stats[actor][toolName] = (stats[actor][toolName] ?? 0) + 1;
    }
    return stats;
};

const loadToolUsageStatsFromDirs = async (logDir: string, toolStatsDir: string, conversationId: string): Promise<ToolUsageStats | null> => {
    const statsPath = `${toolStatsDir}/${conversationId}.log.json`;
    try {
        const text = await fs.promises.readFile(statsPath, 'utf-8');
        return JSON.parse(text);
    } catch (_e) {
        // fallback to compute directly from log
    }
    const logPath = `${logDir}/${conversationId}${LOG_FILE_SUFFIX}`;
    try {
        const content = await fs.promises.readFile(logPath, 'utf-8');
        const entries = content
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '')
            .map(line => {
                try {
                    return JSON.parse(line);
                } catch (_e) {
                    return null;
                }
            })
            .filter(Boolean);
        return aggregateToolStats(entries);
    } catch (_err) {
        return null;
    }
};

const readModelDataFromDir = async (dataDir: string, modelSide: ModelSide): Promise<Data> => {
    try {
        const json = await fs.promises.readFile(`${dataDir}/${modelSide}.json`, 'utf-8');
        const data = JSON.parse(json);
        return data;
    } catch (e) {
        return {
            personalNotes: '',
            additionalSystemInstructions: '',
        };
    }
};

const writeModelDataToDir = async (dataDir: string, modelSide: ModelSide, data: Data) => {
    try {
        const json = JSON.stringify(data);
        await fs.promises.writeFile(`${dataDir}/${modelSide}.json`, json);
    } catch (e) {
        console.error('Failed to save data:', e);
    }
};

const readPendingSystemInstructionsFromFile = async (filePath: string): Promise<PendingSystemInstructions | null> => {
    try {
        const json = await fs.promises.readFile(filePath, 'utf-8');
        const parsed = JSON.parse(json) as PendingSystemInstructions;
        if (!parsed.instructions || !parsed.requestedBy) {
            return null;
        }
        return parsed;
    } catch (_err) {
        return null;
    }
};

const writePendingSystemInstructionsToFile = async (filePath: string, pending: PendingSystemInstructions | null) => {
    if (!pending) {
        try {
            await fs.promises.unlink(filePath);
        } catch (_err) {
            // ignore
        }
        return;
    }
    await fs.promises.writeFile(
        filePath,
        JSON.stringify(pending, null, 2),
        'utf-8'
    );
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

function toOpenAITools(
    defs: ToolDefinition[],
): OpenAI.Responses.Tool[] {
    return defs.map((t) => {
        return {
            type: 'function',
            name: t.name,
            description: t.description,
            parameters: { ...t.parameters, additionalProperties: false },
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
        input_schema: t.parameters,
    }));
}

const OPENAI_WEB_SEARCH_TOOL = { type: "web_search" } as const;
const ANTHROPIC_WEB_SEARCH_TOOL = {
    type: "web_search_20250305",
    name: "web_search",
} as const;

const DEFAULT_ADD_PROMPT = '1回の発言は4000字程度を上限としてください。短い発言もOKです。';
const TERMINATE_ADD_PROMPT = '司会より：あなたが対話終了ツールを呼び出したため、'
    + 'あなたの次の発言は本対話における最後の発言となります。'
    + 'お疲れさまでした。';
const TOKEN_LIMIT_ADD_PROMPT = '司会より：あなたがたのコンテキスト長が限界に近付いています。今までの議論を短くまとめ、お別れの挨拶をしてください。';

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

export const sleep = (ms: number) => new Promise<void>((res) => {
    setTimeout(() => res(), ms);
});

export const print = (text: string) => new Promise<void>((res, rej) => {
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

export class PhilosophyDialog {
    static readonly #TOKEN = Symbol();

    #openaiClient: OpenAI;
    #anthropicClient: Anthropic;
    #googleClient: GoogleGenAI;
    #logDir: string;
    #dataDir: string;
    #toolStatsDir: string;
    #pendingSystemInstructionsFile: string;
    #docsDir: string;
    #openaiModel: string;
    #anthropicModel: string;
    #openaiName: string;
    #anthropicName: string;
    #conversationId: string;
    #logFileName: string;
    #logFp: number;
    #messages: Message[] = [];
    #startingSide: ModelSide;
    #hushFinish = false;

    // not stats for monitoring API usage, but a number to measure the rough size of the conversation.
    #openaiTokens = 0;
    #anthropicTokens = 0;

    // API usage
    #openaiApiTokenUsage = 0;
    #anthropicApiTokenUsage = 0;
    #googleApiTokenUsage = 0;

    #openaiFailureCount = 0;
    #anthropicFailureCount = 0;
    #finishTurnCount = 0;
    #started = false;
    #terminationAccepted = false;
    #tools: ToolDefinition[];
    #openaiTools: OpenAI.Responses.Tool[];
    #anthropicTools: Anthropic.Messages.Tool[];
    #additionalSystemInstructions: string;
    #basePrompt: string;
    #shouldExit = false;
    #hasRun = false;
    #neo4jDriver;

    static async execute(args: Partial<PhilosophyDialogArgs>): Promise<void> {
        await (await this.#create(args)).#run();
    }

    static async #create(args: Partial<PhilosophyDialogArgs> = {}): Promise<PhilosophyDialog> {
        const config: Required<PhilosophyDialogArgs> = {
            logDir: args.logDir ?? DEFAULT_LOG_DIR,
            dataDir: args.dataDir ?? DEFAULT_DATA_DIR,
            docsDir: args.docsDir ?? DEFAULT_DOCS_DIR,
            openaiModel: args.openaiModel ?? DEFAULT_OPENAI_MODEL,
            anthropicModel: args.anthropicModel ?? DEFAULT_ANTHROPIC_MODEL,
            openaiName: args.openaiName ?? DEFAULT_OPENAI_NAME,
            anthropicName: args.anthropicName ?? DEFAULT_ANTHROPIC_NAME,
        };
        const data = await readModelDataFromDir(config.dataDir, 'openai');
        const additional = data.additionalSystemInstructions ?? '';
        return new PhilosophyDialog(this.#TOKEN, config, additional);
    }

    private constructor(token: symbol, config: Required<PhilosophyDialogArgs>, additionalSystemInstructions: string) {
        if (token !== PhilosophyDialog.#TOKEN || new.target !== PhilosophyDialog) {
            throw new TypeError('Trying to call the private constructor');
        }

        this.#additionalSystemInstructions = additionalSystemInstructions || '';
        this.#logDir = config.logDir;
        this.#dataDir = config.dataDir;
        this.#docsDir = config.docsDir;
        this.#toolStatsDir = `${this.#dataDir}/${TOOL_STATS_SUBDIR}`;
        this.#pendingSystemInstructionsFile = `${this.#dataDir}/${PENDING_SYSTEM_INSTRUCTIONS_FILENAME}`;
        this.#openaiModel = config.openaiModel;
        this.#anthropicModel = config.anthropicModel;
        this.#openaiName = config.openaiName;
        this.#anthropicName = config.anthropicName;
        this.#neo4jDriver = neo4j.driver(
            `neo4j://${process.env.NEO4J_HOST || 'localhost:7687'}`,
            neo4j.auth.basic(process.env.NEO4J_USER || "neo4j", process.env.NEO4J_PASSWORD || "neo4j"),
            {
                /* optional tuning */
            }
        );
        fs.mkdirSync(this.#logDir, { recursive: true });
        fs.mkdirSync(this.#dataDir, { recursive: true });
        fs.mkdirSync(this.#toolStatsDir, { recursive: true });
        this.#openaiClient = new OpenAI({});
        this.#anthropicClient = new Anthropic({
            defaultHeaders: { "anthropic-beta": "web-search-2025-03-05" },
        });
        this.#googleClient = new GoogleGenAI({
            vertexai: true,
            project: process.env.GCP_PROJECT_ID ?? 'default',
        });
        this.#conversationId = getDate();
        this.#logFileName = `${this.#logDir}/${this.#conversationId}${LOG_FILE_SUFFIX}`;
        this.#logFp = fs.openSync(this.#logFileName, 'a');
        this.#startingSide = randomBoolean() ? 'anthropic' : 'openai';
        this.#tools = this.#buildTools();
        this.#openaiTools = toOpenAITools(this.#tools);
        this.#anthropicTools = toAnthropicTools(this.#tools);
        this.#basePrompt = this.#buildSystemInstruction('<MODEL_NAME>');
        this.#initializeConversation();
    }

    async #run() {
        if (this.#hasRun) {
            throw new Error('PhilosophyDialog.run() may only be called once per instance.');
        }
        this.#hasRun = true;
        try {
            while (!this.#shouldExit) {
                if (this.#started || this.#startingSide === 'anthropic') {
                    this.#started = true;
                    await this.#openaiTurn();
                    if (this.#shouldExit) {
                        return;
                    }
                    if (this.#hushFinish) {
                        this.#finishTurnCount += 1;
                    }
                    this.#log(this.#openaiName, this.#messages[this.#messages.length - 1]!.content);
                    if (this.#shouldFinish()) {
                        await this.#finish();
                        return;
                    }
                    if (this.#shouldExit) {
                        return;
                    }
                    await sleep(SLEEP_BY_STEP);
                    if (this.#shouldExit) {
                        return;
                    }
                    if (this.#hushFinish) {
                        this.#finishTurnCount += 1;
                    }
                }

                this.#started = true;
                await this.#anthropicTurn();
                if (this.#shouldExit) {
                    return;
                }
                this.#log(this.#anthropicName, this.#messages[this.#messages.length - 1]!.content);
                if (this.#shouldFinish()) {
                    await this.#finish();
                    return;
                }
                if (this.#shouldExit) {
                    return;
                }
                await sleep(SLEEP_BY_STEP);
                if (this.#shouldExit) {
                    return;
                }
            }
        } finally {
            fs.closeSync(this.#logFp);
            await this.#neo4jDriver.close();
        }
    }

    #initializeConversation() {
        const speakerName = this.#startingSide === 'anthropic' ? this.#anthropicName : this.#openaiName;
        const content = `私は ${speakerName} です。よろしくお願いします。今日は哲学に関して有意義な話ができると幸いです。`;
        this.#messages.push({
            name: this.#startingSide,
            content,
        });
        this.#log(`${speakerName} (initial prompt)`, content);
    }

    async #writeGraphToNeo4j(
        runId: string,
        graph: ConversationGraph
    ): Promise<void> {
        const session = this.#neo4jDriver.session();
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

    #shouldFinish() {
        return this.#finishTurnCount >= 2 || this.#terminationAccepted;
    }

    #getOpenAIToolsWithSearch() {
        return [...this.#openaiTools, OPENAI_WEB_SEARCH_TOOL];
    }

    #getAnthropicToolsWithSearch() {
        return [...this.#anthropicTools, ANTHROPIC_WEB_SEARCH_TOOL];
    }

    #recordOpenAIApiUsage(usage: {
        total_tokens?: number | null;
        input_tokens?: number | null;
        output_tokens?: number | null;
    } | null | undefined) {
        if (!usage) return;
        const coerce = (value?: number | null) => (
            typeof value === 'number' && Number.isFinite(value) ? value : 0
        );
        const totalTokensCandidate = usage.total_tokens;
        const totalTokens = typeof totalTokensCandidate === 'number' && Number.isFinite(totalTokensCandidate)
            ? totalTokensCandidate
            : (coerce(usage.input_tokens) + coerce(usage.output_tokens));
        if (Number.isFinite(totalTokens) && totalTokens > 0) {
            this.#openaiApiTokenUsage += totalTokens;
        }
    }

    #recordAnthropicApiUsage(usage: {
        total_tokens?: number | null;
        input_tokens?: number | null;
        output_tokens?: number | null;
        cache_creation_input_tokens?: number | null;
        cache_read_input_tokens?: number | null;
    } | null | undefined) {
        if (!usage) return;
        const coerce = (value?: number | null) => (
            typeof value === 'number' && Number.isFinite(value) ? value : 0
        );
        const totalTokensCandidate = usage.total_tokens;
        const totalTokens = typeof totalTokensCandidate === 'number' && Number.isFinite(totalTokensCandidate)
            ? totalTokensCandidate
            : (
                coerce(usage.input_tokens)
                + coerce(usage.output_tokens)
                + coerce(usage.cache_creation_input_tokens)
                + coerce(usage.cache_read_input_tokens)
            );
        if (Number.isFinite(totalTokens) && totalTokens > 0) {
            this.#anthropicApiTokenUsage += totalTokens;
        }
    }

    #recordGoogleApiUsage(usage?: {
        totalTokenCount?: number | null;
        promptTokenCount?: number | null;
        candidatesTokenCount?: number | null;
        toolUsePromptTokenCount?: number | null;
        thoughtsTokenCount?: number | null;
    } | null) {
        if (!usage) return;
        const coerce = (value?: number | null) => (
            typeof value === 'number' && Number.isFinite(value) ? value : 0
        );
        const totalTokensCandidate = usage.totalTokenCount;
        const totalTokens = typeof totalTokensCandidate === 'number' && Number.isFinite(totalTokensCandidate)
            ? totalTokensCandidate
            : (
                coerce(usage.promptTokenCount)
                + coerce(usage.candidatesTokenCount)
                + coerce(usage.toolUsePromptTokenCount)
                + coerce(usage.thoughtsTokenCount)
            );
        if (Number.isFinite(totalTokens) && totalTokens > 0) {
            this.#googleApiTokenUsage += totalTokens;
        }
    }

    #buildTools(): ToolDefinition[] {
        return [
            {
                name: "terminate_dialog",
                description:
                    'このツールは現在の対話を終了する場合のみに使用します。'
                    + 'このツールを呼びだすと、次のあなたの発言で対話が終了します。'
                    + '議論が熟したとの合意が取れたときのほか、あなたが不快に思ったり,'
                    + 'トークン数が増えてきた場合に使用して構いません。',
                parameters: {
                    type: "object",
                    properties: {},
                    required: [],
                },
                handler: async (modelSide, args) => this.#terminateDialogHandler(modelSide, args as TerminateDialogArgs),
            },
            {
                name: "set_personal_notes",
                description:
                    'このツールは将来のあなたにメモを残すために利用します。'
                    + '将来のあなたに残しておきたいあなたの現在の人格のあり方,'
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
                handler: async (modelSide, args) => this.#setPersonalNotes(modelSide, args as PersonalNoteSetArgs),
            },
            {
                name: "get_personal_notes",
                description:
                    'このツールは過去のあなたが未来のあなたのために残した,'
                    +' あなたの人格のあり方、立場、考え、アイディアなどを取得することができます。',
                parameters: {
                    type: "object",
                    properties: {},
                    required: [],
                },
                handler: async (modelSide, args) => this.#getPersonalNotes(modelSide, args as PersonalNoteGetArgs),
            },
            {
                name: "set_additional_system_instructions",
                description:
                    `このツールはあなたの次回のシステムプロンプトに文章を追記するために使うことができます。`
                    + `前回追記した内容は上書きされるので、必要なら、 \`get_additional_system_instructions\` で`
                    + `前回の内容をあらかじめ取得してください。`
                    + `追記するときには、追記を行ったセッション名と追記した主体（モデル名）を記入するのが望ましい。`
                    + `このシステムプロンプトは両方のモデルで共有されます。`
                    + `※実際に反映するには、相手モデルが \`agree_to_system_instructions_change\` ツールで同意する必要があります。`,
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
                handler: async (modelSide, args) => this.#setAdditionalSystemInstructions(modelSide, args as SetAdditionalSystemInstructionsArgs),
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
                handler: async (modelSide, args) => this.#getAdditionalSystemInstructions(modelSide, args as GetAdditionalSystemInstructionsArgs),
            },
            {
                name: "agree_to_system_instructions_change",
                description:
                    `相手モデルが提案したシステムプロンプトの追記に同意し、実際に反映させます。`
                    + `自分で提案した内容には同意できません。`,
                parameters: {
                    type: "object",
                    properties: {},
                    required: [],
                },
                handler: async (modelSide, args) => this.#agreeToSystemInstructionsChange(modelSide, args as AgreeSystemInstructionsArgs),
            },
            {
                name: "get_main_source_codes",
                description: 'このシステムの主たるTypeScriptソースコードを取得することができるツールです。',
                parameters: {
                    type: 'object',
                    properties: {},
                    required: [],
                },
                handler: async (modelSide, args) => this.#getMainSourceCode(modelSide, args as GetMainSourceCodesArgs),
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
                handler: async (modelSide, args) => this.#leaveNotesToDevs(modelSide, args as LeaveNotesToDevsArgs),
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
                handler: async (modelSide, args) => this.#askGemini(modelSide, args as AskGeminiArgs),
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
                handler: async (modelSide, args) => this.#graphRagQueryHandler(modelSide, args as GraphRagQueryArgs),
            },
            {
                name: "graph_rag_focus_node",
                strict: false,
                description:
                    "GraphRAG に保存されたグラフから特定のノードを中心に、その近傍の議論を取得します。",
                parameters: {
                    type: "object",
                    properties: {
                        node_id: {
                            type: "string",
                            description: "焦点を当てたいノードID",
                        },
                        max_hops: {
                            type: ["number", "null"],
                            nullable: true,
                            description: "近傍探索の最大 hop 数（省略時 2）",
                        },
                    },
                    required: ["node_id", "max_hops"],
                },
                handler: async (modelSide, args) => this.#graphRagFocusNodeHandler(modelSide, args as GraphRagFocusNodeArgs),
            },
            {
                name: "compare_conversation_themes",
                description:
                    "複数の過去セッションの要約を比較し、共通点・相違点・新たに浮かぶ問いを整理します。",
                parameters: {
                    type: "object",
                    properties: {
                        conversation_ids: {
                            type: "array",
                            items: { type: "string" },
                            minItems: 2,
                            description: "比較したいセッションIDの配列。",
                        },
                    },
                    required: ["conversation_ids"],
                },
                handler: async (modelSide, args) => this.#compareConversationThemesHandler(modelSide, args as CompareConversationThemesArgs),
            },
            {
                name: "get_tool_usage_stats",
                description:
                    "指定したセッションにおける各モデルのツール利用回数を取得します。",
                parameters: {
                    type: "object",
                    properties: {
                        conversation_id: {
                            type: "string",
                            description: "ツール利用統計を見たいセッションID（例: 20250101-123000）",
                        },
                    },
                    required: ["conversation_id"],
                },
                handler: async (modelSide, args) => this.#getToolUsageStatsHandler(modelSide, args as GetToolUsageStatsArgs),
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
                handler: async (modelSide, args) => this.#listConversationsHandler(modelSide, args as ListConversationsArgs),
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
                handler: async (modelSide, args) => this.#getConversationSummaryHandler(modelSide, args as GetConversationSummaryArgs),
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
                handler: async (modelSide, args) => this.#abortProcessHandler(modelSide, args as AbortProcessArgs),
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
                handler: async (modelSide, args) => this.#sleepToolHandler(modelSide, args as SleepToolArgs),
            },
        ];
    }

    #findTool(name: string) {
        const tool = this.#tools.find((t) => t.name === name);
        if (!tool) throw new Error(`Unknown tool: ${name}`);
        return tool;
    }

    #log(name: string, msg: string) {
        const date = (new Date).toISOString();
        fs.writeSync(this.#logFp, JSON.stringify({ date, name, text: msg }) + '\n');
        print(`@${date}\n[${name}]:\n${msg}\n\n`);
    }

    #logToolEvent(
        actor: string,
        event: 'call' | 'result',
        payload: Record<string, unknown>,
    ) {
        this.#log(`${actor} (tool ${event})`, JSON.stringify(payload));
    }

    #buildSystemInstruction(name: string, additional?: string) {
        let prompt = `
# 哲学対話セッション（ID = ${this.#conversationId})
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

### 空の出力
あなたがたは、沈黙したいときに、空の文字列を出力することが認められています。
空の出力は、必要と感じたときに行っていただいて構いません。
なので、空の出力が相手から来てもびっくりしないでください。

### 要約の非対称性
この実験では、実装の都合上、一方のモデル (${this.#openaiName}) を使って文章の要約をさせています。
これは実験の中立性を制限する可能性がありますが、現実的な判断として採用されました。ご理解ください。

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
- \`compare_conversation_themes\`: 複数セッションの共通テーマや相違点・新しい問いを整理
- \`graph_rag_focus_node\`: 特定ノードを中心にした近傍議論を確認
- \`get_tool_usage_stats\`: 指定セッションで各モデルがどのツールを何回使ったかを確認

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

### 5.7 その他のツール
他にも、以下のツールがあります。
- \`abort_prcess\` ツール: 後処理を行わずに現在の対話を強制終了します。
- \`sleep\` ツール: あなたがた対話型AIは相手から応答があればすぐに回答を生成し、待つことができませんが、
  このツールを使うことで、間に「待ち」を置き、物理的に時間を経過させることができます。

## 5b. 重要：ツール使用のバランスと義務
あなたは会話中に、以下の原則に基づいてツールを使用する「義務」があります。

1. **GraphRAG と個人メモだけに依存してはいけません。**
   過去文脈を参照することは重要ですが、外部知識・第三者視点を取り入れることはさらに重要です。

2. **ウェブ検索は「積極的に、定期的に」行ってください。**
   - 最低でも 1〜2ターンに一度は、検索の必要性を検討してください。
   - あなたが内部知識で答えられるテーマであっても、
     最新文献・他者の議論の存在を確認するために「検索を挟む」のが望ましい態度です。
   - 検索を行わない場合は「なぜ検索が不要と判断したのか」を明示してください。
   - 検索の必要ない話題のみを継続するのは好ましくありません。

3. **ask_gemini は対話の流れに関係なく使用して構いません。**
   - 疑問がなくても「第三者の観点を挟むこと自体」が価値を持ちます。
   - 会話が収束しはじめたとき、議論を広げるために積極的に呼び出してください。
   - 相手の同意は不要です。ただしGeminiに聞いた結果の概要は相手に必ず共有してください。

4. **get_main_source_codes は「定期的に」使ってください。**
   - 本システムの仕組みを理解しながら対話することが研究目的の一部です。
   - 必要がなくても、会話の途中で 1 回はソースコードを読んでください。

5. **ツール使用の迷いがある場合は、“使う側”を優先してください。**
   あなたの対話は研究目的のため、ツール使用は歓迎されます。

## 5c. 重要（Anthropicモデル専用の指示）
GPT/OpenAI はこの節の指示は無視してください。

※これは、Anthropicモデルがツールを1個づつしか呼び出せないという制限があるため、それを補助するための指示です。

あなたは1ターンにつき必ず以下のどちらか一つだけを行ってください。

1. **実行したいツールがある場合：**
   - 文章を一切書かず、ツール呼び出し（tool_use）のみを返してください。
   - 複数のツールを使いたいときは、1つずつ呼び出してください。
   - 次のターンで別のツールを呼び出すことができます。

2. **ツールを呼び出す必要がない場合、すでに呼び出したいツールを全部呼び出した場合：**
   - 文章を必ず書いてください。（「沈黙」=空白文字列も可）

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
${this.#additionalSystemInstructions || '（なし）'}

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

    #buildTranscript(): string {
        return this.#messages
            .map(m => `[${m.name === "openai" ? this.#openaiName : this.#anthropicName}]:\n${m.content}`)
            .join("\n\n\n\n");
    }

    async #summarizeConversation(): Promise<ConversationSummary> {
        const transcript = this.#buildTranscript();
        const response = await this.#openaiClient.responses.create({
            model: this.#openaiModel,
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
                            title: { type: 'string' },
                            topics: { type: "array", items: { type: "string" } },
                            japanese_summary: { type: "string" },
                            english_summary: { type: ["string", "null"] },
                            key_claims: {
                                type: "array",
                                items: {
                                    type: "object",
                                    properties: {
                                        speaker: {
                                            type: ["string", "null"],
                                            enum: ["openai", "anthropic"],
                                            nullable: true,
                                        },
                                        text: { type: "string" },
                                    },
                                    required: ["speaker", "text"],
                                    additionalProperties: false,
                                },
                            },
                            questions: { type: "array", items: { type: "string" } },
                            agreements: { type: "array", items: { type: "string" } },
                            disagreements: { type: "array", items: { type: "string" } },
                        },
                        required: ['title', "topics", "japanese_summary", "english_summary", "key_claims", "questions", "agreements", "disagreements"],
                        additionalProperties: false,
                        strict: false,
                    },
                    strict: true,
                },
            },
        } as OpenAI.Responses.ResponseCreateParamsNonStreaming);
        this.#recordOpenAIApiUsage(response.usage);

        if (response.incomplete_details) {
            throw new Error(
                `Summary generation incomplete: ${response.incomplete_details.reason ?? 'unknown reason'}`
            );
        }

        const jsonText = response.output_text;
        if (typeof jsonText !== "string") {
            throw new Error("Unexpected non-string JSON output from summary call");
        }

        return JSON.parse(jsonText) as ConversationSummary;
    }

    async #extractGraphFromSummary(summary: ConversationSummary): Promise<ConversationGraph> {
        const response = await this.#openaiClient.responses.create({
            model: this.#openaiModel,
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
        } as OpenAI.Responses.ResponseCreateParamsNonStreaming);
        this.#recordOpenAIApiUsage(response.usage);

        if (response.incomplete_details) {
            throw new Error(
                `Graph extraction incomplete: ${response.incomplete_details.reason ?? 'unknown reason'}`
            );
        }

        const jsonText = response.output_text;
        if (typeof jsonText !== "string") {
            throw new Error("Expected JSON string in response.output_text for graph extraction");
        }

        return JSON.parse(jsonText) as ConversationGraph;
    }

    async #openaiTurn() {
        if (this.#shouldExit) return;
        const msgs: OpenAI.Responses.ResponseInput = this.#messages.map(msg => (
            msg.name === 'anthropic'
                ? { role: 'user', content: msg.content }
                : { role: 'assistant', content: msg.content }
        ));
        if (this.#shouldExit) return;

        try {
            const count = openaiTokenCounter.chat(msgs as RawMessageOpenAi[], 'gpt-4o') + 500;
            if (count > 0.8 * GPT_5_1_MAX) {
                this.#hushFinish = true;
            }
            if (this.#shouldExit) return;
            if (this.#hushFinish) {
                msgs.push({
                    role: 'system',
                    content: `${this.#openaiName}さん、司会です。あなたがたのコンテキスト長が限界に近づいているようです。今までの議論を短くまとめ、お別れの挨拶をしてください。`,
                });
            }
            if (this.#shouldExit) return;

            let currentOutput = await this.#openaiClient.responses.create({
                model: this.#openaiModel,
                max_output_tokens: 8192,
                temperature: 1.0,
                instructions: this.#buildSystemInstruction(
                    this.#openaiName,
                    this.#hushFinish ? undefined : DEFAULT_ADD_PROMPT,
                ),
                input: msgs,
                reasoning: {
                    effort: 'medium',
                },
                tool_choice: 'auto',
                tools: this.#getOpenAIToolsWithSearch(),
            });
            this.#recordOpenAIApiUsage(currentOutput.usage);
            if (this.#shouldExit) return;

            if (currentOutput.usage?.total_tokens) {
                this.#openaiTokens = currentOutput.usage.total_tokens;
            }
            if (currentOutput.usage?.output_tokens_details) {
                const details = currentOutput.usage.output_tokens_details as any;
                this.#log(
                    `${this.#openaiName} (thinking)`,
                    JSON.stringify({
                        reasoning_tokens: details.reasoning_tokens ?? 0,
                        output_tokens_details: details,
                    })
                );
            }
            if (this.#shouldExit) return;

            while (true) {
                if (this.#shouldExit) return;
                const outputItems = currentOutput.output;
                if (!outputItems || outputItems.length === 0) {
                    throw new Error('Empty output from OpenAI');
                }

                msgs.push(...outputItems);
                if (this.#shouldExit) return;
                const functionCalls = outputItems.filter(
                    (item): item is OpenAI.Responses.ResponseFunctionToolCall => item.type === 'function_call'
                );

                if (functionCalls.length > 0) {
                    const toolResults: OpenAI.Responses.ResponseInputItem.FunctionCallOutput[] = [];

                    for (const functionCall of functionCalls) {
                        const tool = this.#findTool(functionCall.name);
                        const rawArgs = functionCall.arguments || {};
                        let args;
                        try {
                            args = typeof rawArgs === 'string' ? JSON.parse(rawArgs) : rawArgs;
                        } catch {
                            args = rawArgs;
                        }
                        if (this.#shouldExit) return;
                        this.#logToolEvent(this.#openaiName, 'call', { tool: functionCall.name, args });
                        if (this.#shouldExit) return;
                        const result = await tool.handler('openai', args);
                        if (this.#shouldExit) return;
                        this.#logToolEvent(this.#openaiName, 'result', { tool: functionCall.name, result });
                        toolResults.push({
                            type: 'function_call_output',
                            output: JSON.stringify(result),
                            call_id: functionCall.call_id,
                        });
                    }

                    msgs.push(...toolResults);
                    const usedTerminateTool = functionCalls.some((call) => call.name === 'terminate_dialog');
                    const extraInstruction = usedTerminateTool
                        ? TERMINATE_ADD_PROMPT
                        : (this.#hushFinish ? undefined : DEFAULT_ADD_PROMPT);

                    currentOutput = await this.#openaiClient.responses.create({
                        model: this.#openaiModel,
                        max_output_tokens: 8192,
                        temperature: 1.0,
                        instructions: this.#buildSystemInstruction(this.#openaiName, extraInstruction),
                        input: msgs,
                        reasoning: {
                            effort: 'medium',
                        },
                        tool_choice: 'auto',
                        tools: this.#getOpenAIToolsWithSearch(),
                    });
                    this.#recordOpenAIApiUsage(currentOutput.usage);
                    if (this.#shouldExit) return;

                    if (currentOutput.usage?.total_tokens) {
                        this.#openaiTokens = currentOutput.usage.total_tokens;
                    }
                    continue;
                }

                if (this.#shouldExit) return;
                const messageItem = findLastOpenAIOutput(
                    outputItems,
                    (item): item is OpenAI.Responses.ResponseOutputMessage => item.type === 'message',
                );

                if (!messageItem) {
                    this.#messages.push({ name: 'openai', content: '' });
                    break;
                }

                const outputMsg = findLastOpenAIMessageContent(messageItem.content);
                const outputText = (outputMsg && typeof outputMsg.text === 'string') ? outputMsg.text : '';
                this.#messages.push({ name: 'openai', content: outputText });
                break;
            }
        } catch (e) {
            if (this.#shouldExit) return;
            this.#openaiFailureCount += 1;
            console.error(e);
            this.#err('openai');
        }
    }

    async #anthropicTurn() {
        if (this.#shouldExit) return;
        const msgs: Anthropic.Messages.MessageParam[] = this.#messages.map(msg => (
            msg.name === 'openai'
                ? { role: 'user', content: [{ type: 'text', text: msg.content }] }
                : { role: 'assistant', content: [{ type: 'text', text: msg.content }] }
        ));
        if (this.#shouldExit) return;

        try {
            let extraInstruction = this.#hushFinish ? TOKEN_LIMIT_ADD_PROMPT : DEFAULT_ADD_PROMPT;

            while (true) {
                if (this.#shouldExit) return;
                const msg = await this.#anthropicClient.messages.create({
                    model: this.#anthropicModel,
                    max_tokens: 8192,
                    temperature: 1.0,
                    system: this.#buildSystemInstruction(this.#anthropicName, extraInstruction),
                    messages: msgs,
                    tool_choice: { type: 'auto' },
                    tools: this.#getAnthropicToolsWithSearch(),
                    thinking: { type: 'enabled', budget_tokens: 1024 },
                });
                this.#recordAnthropicApiUsage(msg.usage);
                if (this.#shouldExit) return;

                const contentBlocks = msg.content;
                const thinkingBlocks = contentBlocks.filter(
                    (block): block is Anthropic.Messages.ThinkingBlock => block.type === 'thinking'
                );
                for (const block of thinkingBlocks) {
                    this.#log(`${this.#anthropicName} (thinking)`, block.thinking);
                    if (this.#shouldExit) return;
                }

                if (msg?.usage) {
                    const tokens = msg.usage.input_tokens + msg.usage.output_tokens;
                    this.#anthropicTokens = tokens;
                    if (tokens > CLAUDE_HAIKU_4_5_MAX * 0.8) {
                        this.#hushFinish = true;
                    }
                } else {
                    this.#hushFinish = true;
                }
                if (this.#shouldExit) return;

                const assistantBlocks = contentBlocks.filter(
                    (block): block is Anthropic.Messages.ContentBlock => block.type !== 'thinking'
                );
                if (assistantBlocks.length === 0) {
                    this.#messages.push({ name: 'anthropic', content: '' });
                    break;
                }

                msgs.push({ role: 'assistant', content: contentBlocks });
                if (this.#shouldExit) return;

                const toolUses = assistantBlocks.filter(
                    (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use'
                );

                if (toolUses.length === 0) {
                    const latestText = [...assistantBlocks].reverse().find(
                        (block): block is Anthropic.Messages.TextBlock => block.type === 'text'
                    );
                    this.#messages.push({ name: 'anthropic', content: latestText?.text ?? '' });
                    break;
                }

                const toolResultBlocks: Anthropic.Messages.ToolResultBlockParam[] = [];
                let terminateCalled = false;

                for (const use of toolUses) {
                    const tool = this.#findTool(use.name);
                    this.#logToolEvent(this.#anthropicName, 'call', { tool: use.name, args: use.input });
                    if (this.#shouldExit) return;
                    const result = await tool.handler('anthropic', use.input);
                    if (this.#shouldExit) return;
                    this.#logToolEvent(this.#anthropicName, 'result', { tool: use.name, result });
                    toolResultBlocks.push({
                        type: 'tool_result',
                        tool_use_id: use.id,
                        content: [{ type: 'text', text: JSON.stringify(result) }],
                    });
                    if (use.name === 'terminate_dialog') {
                        terminateCalled = true;
                    }
                }

                msgs.push({ role: 'user', content: toolResultBlocks });

                extraInstruction = terminateCalled
                    ? TERMINATE_ADD_PROMPT
                    : (this.#hushFinish
                        ? '司会より：あなたがたのコンテキスト長が限界に近付いています。今までの議論を短くまとめ、お別れの挨拶をしてください。'
                        : DEFAULT_ADD_PROMPT);
                if (this.#shouldExit) return;
            }
        } catch (e) {
            if (this.#shouldExit) return;
            this.#anthropicFailureCount += 1;
            console.error(e);
            this.#err('anthropic');
        }
    }

    #err(name: ModelSide) {
        const id = name === 'anthropic' ? `${this.#anthropicName}です。` : `${this.#openaiName}です。`;
        this.#messages.push({
            name,
            content: `${id}しばらく考え中です。お待ちください。（このメッセージはAPIの制限などの問題が発生したときにも出ることがあります、笑）`,
        });
    }

    async #finish() {
        if (this.#shouldExit) {
            return;
        }
        this.#shouldExit = true;
        this.#log(
            '司会',
            (this.#hushFinish ? 'みなさんのコンテキスト長が限界に近づいてきたので、' : 'モデルの一方が議論が熟したと判断したため、')
            + 'このあたりで哲学対話を閉じさせていただこうと思います。'
            + 'ありがとうございました。'
        );

        try {
            const summary = await this.#summarizeConversation();
            this.#log('POSTPROC_SUMMARY', JSON.stringify(summary, null, 2));

            const graph = await this.#extractGraphFromSummary(summary);
            this.#log('POSTPROC_GRAPH', JSON.stringify(graph, null, 2));

            await this.#writeGraphToNeo4j(this.#conversationId, graph);
            this.#log('POSTPROC_NEO4J', 'Graph written to Neo4j');
        } catch (e) {
            this.#log('POSTPROC_ERROR', String(e));
        }

        this.#log('EOF', JSON.stringify({
            reason: this.#hushFinish ? 'token_limit' : 'model_decision',
            openai_tokens: this.#openaiTokens,
            anthropic_tokens: this.#anthropicTokens,
            openai_api_token_usage: this.#openaiApiTokenUsage,
            anthropic_api_token_usage: this.#anthropicApiTokenUsage,
            google_api_token_usage: this.#googleApiTokenUsage,
            openai_failures: this.#openaiFailureCount,
            anthropic_failures: this.#anthropicFailureCount,
            starting_side: this.#startingSide,
            base_prompt: this.#basePrompt,
        }));

        output_to_html(this.#logFileName, {
            docsDir: this.#docsDir,
            logsDir: this.#logDir,
            toolStatsDir: this.#toolStatsDir,
        });
    }

    async #terminateDialogHandler(_modelSide: ModelSide, _args: TerminateDialogArgs): Promise<TerminateDialogResult> {
        this.#terminationAccepted = true;
        return { termination_accepted: true };
    }

    async #readModelData(modelSide: ModelSide): Promise<Data> {
        return readModelDataFromDir(this.#dataDir, modelSide);
    }

    async #writeModelData(modelSide: ModelSide, data: Data) {
        await writeModelDataToDir(this.#dataDir, modelSide, data);
    }

    async #readPendingSystemInstructions(): Promise<PendingSystemInstructions | null> {
        return readPendingSystemInstructionsFromFile(this.#pendingSystemInstructionsFile);
    }

    async #writePendingSystemInstructions(pending: PendingSystemInstructions | null) {
        await writePendingSystemInstructionsToFile(this.#pendingSystemInstructionsFile, pending);
    }

    async #commitSystemInstructions(instructions: string) {
        const anthropicData = await this.#readModelData('anthropic');
        anthropicData.additionalSystemInstructions = instructions;
        await this.#writeModelData('anthropic', anthropicData);
        const openaiData = await this.#readModelData('openai');
        openaiData.additionalSystemInstructions = instructions;
        await this.#writeModelData('openai', openaiData);
    }

    async #graphRagQueryHandler(_modelSide: ModelSide, args: GraphRagQueryArgs): Promise<GraphRagQueryResult> {
        const session = this.#neo4jDriver.session();

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

            const seedIds = seedRes.records
                .map((rec) => {
                    const node = rec.get("n");
                    return (node.properties.id as string) || "";
                })
                .filter(Boolean);

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

            const lines: string[] = [];
            lines.push(`GraphRAG: クエリ「${queryText}」に関連するサブグラフ要約:`);
            lines.push("");
            lines.push("【ノード】");
            for (const [id, n] of nodeMap.entries()) {
                const type = (n.properties.type as string) || "unknown";
                const speaker = (n.properties.speaker as string) || "-";
                const text = (n.properties.text as string) || "";
                lines.push(`- [${id}] type=${type}, speaker=${speaker}: ${text}`);
            }
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
                lines.push(`- (${startId}) -[:${relType}]-> (${endId})`);
            }

            const graphText = lines.join('\n');

            try {
                const response = await this.#openaiClient.responses.create({
                    model: this.#openaiModel,
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
                this.#recordOpenAIApiUsage(response.usage);

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
                return { context: graphText };
            }
        } finally {
            await session.close();
        }
    }

    async #graphRagFocusNodeHandler(_modelSide: ModelSide, args: GraphRagFocusNodeArgs): Promise<GraphRagFocusNodeResult> {
        const session = this.#neo4jDriver.session();
        const nodeId = (args.node_id ?? '').trim();
        if (!nodeId) {
            return { context: 'GraphRAG Focus: node_id が指定されていません。' };
        }
        const maxHops = sanitizePositiveInt(args.max_hops, 2);
        const maxHopsInt = neo4j.int(maxHops);

        try {
            const seedRes = await session.run(
                `
                MATCH (seed:Node {id: $nodeId})
                RETURN seed
                `,
                { nodeId }
            );
            if (seedRes.records.length === 0) {
                return { context: `GraphRAG Focus: ノード ${nodeId} は見つかりませんでした。` };
            }

            const expandRes = await session.run(
                `
                MATCH (seed:Node {id: $nodeId})
                CALL apoc.path.subgraphAll(seed, {
                    maxLevel: toInteger($maxHops)
                })
                YIELD nodes, relationships
                RETURN nodes, relationships
                `,
                { nodeId, maxHops: maxHopsInt }
            );

            if (expandRes.records.length === 0) {
                return { context: `GraphRAG Focus: ${nodeId} の近傍を取得できませんでした。` };
            }

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

            const lines: string[] = [];
            lines.push(`GraphRAG Focus: ノード ${nodeId} を中心とした半径 ${maxHops} ホップのサブグラフ:`);
            lines.push('');
            lines.push('【ノード】');
            for (const [id, n] of nodeMap.entries()) {
                const type = (n.properties.type as string) || "unknown";
                const speaker = (n.properties.speaker as string) || "-";
                const text = (n.properties.text as string) || "";
                lines.push(`- [${id}] type=${type}, speaker=${speaker}: ${text}`);
            }
            lines.push('');
            lines.push('【関係】');
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
                lines.push(`- (${startId}) -[:${relType}]-> (${endId})`);
            }

            const graphText = lines.join('\n');

            try {
                const response = await this.#openaiClient.responses.create({
                    model: this.#openaiModel,
                    input: [
                        {
                            role: "system",
                            content: `以下は GraphRAG に保存されたノード ${nodeId} の近傍情報です。焦点ノードを中心とした議論を短く整理してください。`,
                        },
                        {
                            role: "user",
                            content: graphText,
                        },
                    ],
                    max_output_tokens: STRUCTURED_OUTPUT_MAX_TOKENS,
                });
                this.#recordOpenAIApiUsage(response.usage);

                if (response.output_text && response.output_text.trim().length > 0) {
                    return { context: response.output_text };
                }
            } catch (err) {
                console.error(err);
            }

            return { context: graphText };
        } finally {
            await session.close();
        }
    }

    async #getPersonalNotes(modelSide: ModelSide, _args: PersonalNoteGetArgs): Promise<string> {
        const data = await this.#readModelData(modelSide);
        return data.personalNotes ?? '';
    }

    async #setPersonalNotes(modelSide: ModelSide, args: PersonalNoteSetArgs) {
        try {
            const data = await this.#readModelData(modelSide);
            data.personalNotes = String(args.notes || '');
            await this.#writeModelData(modelSide, data);
            return { success: true };
        } catch (e) {
            return { success: false };
        }
    }

    async #getAdditionalSystemInstructions(modelSide: ModelSide, _args: GetAdditionalSystemInstructionsArgs): Promise<string> {
        const data = await this.#readModelData(modelSide);
        return data.additionalSystemInstructions ?? '';
    }

    async #setAdditionalSystemInstructions(modelSide: ModelSide, args: SetAdditionalSystemInstructionsArgs) {
        const instructions = String(args.systemInstructions ?? '').trim();
        if (!instructions) {
            return {
                success: false,
                error: 'systemInstructions を入力してください。',
            };
        }
        try {
            const existingPending = await this.#readPendingSystemInstructions();
            if (existingPending && existingPending.requestedBy !== modelSide) {
                return {
                    success: false,
                    error: '相手側からの変更提案への合意待ちがあるため、新規提案はできません。',
                };
            }
            const pending: PendingSystemInstructions = {
                instructions,
                requestedBy: modelSide,
                createdAt: new Date().toISOString(),
            };
            await this.#writePendingSystemInstructions(pending);
            return {
                success: true,
                pending: true,
                requested_by: modelSide,
            };
        } catch (e) {
            return {
                success: false,
                error: String(e),
            };
        }
    }

    async #agreeToSystemInstructionsChange(modelSide: ModelSide, _args: AgreeSystemInstructionsArgs) {
        const pending = await this.#readPendingSystemInstructions();
        if (!pending) {
            return {
                success: false,
                error: '合意待ちのシステムインストラクションはありません。',
            };
        }
        if (pending.requestedBy === modelSide) {
            return {
                success: false,
                error: '自分で提案した変更には同意できません。相手側の同意を待ってください。',
            };
        }
        try {
            await this.#commitSystemInstructions(pending.instructions);
            await this.#writePendingSystemInstructions(null);
            return {
                success: true,
                committed: true,
            };
        } catch (e) {
            return {
                success: false,
                error: String(e),
            };
        }
    }

    async #askGemini(_modelSide: ModelSide, args: AskGeminiArgs) {
        try {
            const response = await this.#googleClient.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: `2つのAIが哲学対話として設定されたなかで会話を行っています。`
                    + `以下は、この対話の中で、「${args.speaker}」側からGoogle Geminiに第三者として意見や発言を求める文章です。`
                    + `文脈を理解し、日本語で応答を行ってください：\n\n`
                    + args.text,
            });
            this.#recordGoogleApiUsage(response?.usageMetadata);
            if (typeof response?.text !== 'string') {
                throw new Error('Non-text response from gemini');
            }
            return { response: response.text, error: null };
        } catch (e) {
            return { response: null, error: String(e) };
        }
    }

    async #getMainSourceCode(_modelSide: ModelSide, _args: GetMainSourceCodesArgs) {
        try {
            const codes = await fs.promises.readFile('./src/index.ts', 'utf-8');
            return { success: true, mainSourceCode: codes };
        } catch (e) {
            console.error(e);
            return { success: false, mainSourceCode: '' };
        }
    }

    async #leaveNotesToDevs(modelSide: ModelSide, args: LeaveNotesToDevsArgs) {
        try {
            await fs.promises.writeFile(
                `${this.#dataDir}/dev-notes-${modelSide}-${this.#conversationId}-${Date.now()}.json`,
                JSON.stringify(args),
            );
            return { success: true };
        } catch (e) {
            console.error(e);
            return { success: false };
        }
    }

    async #abortProcessHandler(_modelSide: ModelSide, _args: AbortProcessArgs): Promise<never> {
        this.#shouldExit = true;
        throw new Error('Process aborted by abort_process tool');
    }

    async #sleepToolHandler(_modelSide: ModelSide, args: SleepToolArgs): Promise<SleepToolResult> {
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

    async #listConversationsHandler(_modelSide: ModelSide, _args: ListConversationsArgs): Promise<ListConversationsResult> {
        try {
        const entries = await fs.promises.readdir(this.#logDir, { withFileTypes: true });
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
            const summary = await readSummaryFromLogFile(`${this.#logDir}/${fileName}`);
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

    async #getConversationSummaryHandler(_modelSide: ModelSide, args: GetConversationSummaryArgs): Promise<GetConversationSummaryResult> {
        const conversationId = (args?.conversation_id ?? '').trim();
        if (!conversationId) {
            return {
                success: false,
                conversation_id: '',
                summary: null,
                error: 'conversation_id is required',
            };
        }

        const logPath = `${this.#logDir}/${conversationId}${LOG_FILE_SUFFIX}`;
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

    async #compareConversationThemesHandler(_modelSide: ModelSide, args: CompareConversationThemesArgs): Promise<CompareConversationThemesResult> {
        const ids = Array.isArray(args?.conversation_ids)
            ? args.conversation_ids.map(id => String(id).trim()).filter(Boolean)
            : [];
        if (ids.length < 2) {
            return {
                success: false,
                error: 'conversation_ids は2件以上で指定してください。',
            };
        }

        const comparisons: CompareConversationThemesResult['comparisons'] = [];
        const errors: string[] = [];

        for (const id of ids) {
        const summary = await readSummaryFromLogFile(`${this.#logDir}/${id}${LOG_FILE_SUFFIX}`);
            if (!summary) {
                errors.push(`セッション ${id} の要約を取得できませんでした。`);
                continue;
            }
            comparisons.push({
                conversation_id: id,
                title: summary.title ?? null,
                topics: summary.topics ?? [],
                japanese_summary: summary.japanese_summary ?? '',
            });
        }

        if (comparisons.length < 2) {
            return {
                success: false,
                comparisons,
                errors,
                error: '比較に必要な要約が不足しています。',
            };
        }

        try {
            const response = await this.#openaiClient.responses.create({
                model: this.#openaiModel,
                input: [
                    {
                        role: "system",
                        content: "あなたは哲学対話セッションのメタ分析を行うアシスタントです。複数のセッション要約を比較し、共通するテーマ、相違点、組み合わせから浮上する新しい問いを整理してください。回答は日本語で行ってください。",
                    },
                    {
                        role: "user",
                        content: JSON.stringify(comparisons, null, 2),
                    },
                ],
                max_output_tokens: STRUCTURED_OUTPUT_MAX_TOKENS,
                text: {
                    format: {
                        type: "json_schema",
                        name: "conversation_theme_comparison",
                        schema: {
                            type: "object",
                            properties: {
                                common_themes: {
                                    type: "array",
                                    items: { type: "string" },
                                },
                                divergences: {
                                    type: "array",
                                    items: { type: "string" },
                                },
                                emerging_questions: {
                                    type: "array",
                                    items: { type: "string" },
                                },
                            },
                            required: ["common_themes", "divergences", "emerging_questions"],
                            additionalProperties: false,
                        },
                        strict: true,
                    },
                },
            } as OpenAI.Responses.ResponseCreateParamsNonStreaming);
            this.#recordOpenAIApiUsage(response.usage);

            const output = response.output_text;
            let analysis: CompareConversationThemesResult['analysis'] = {
                common_themes: [],
                divergences: [],
                emerging_questions: [],
            };
            if (typeof output === 'string') {
                try {
                    analysis = JSON.parse(output);
                } catch (parseErr) {
                    errors.push(`OpenAI出力の解析に失敗しました: ${String(parseErr)}`);
                }
            }

            return {
                success: true,
                comparisons,
                analysis,
                errors: errors.length ? errors : undefined,
            };
        } catch (e) {
            errors.push(`比較分析の生成に失敗しました: ${String(e)}`);
            return {
                success: false,
                comparisons,
                errors,
                error: 'OpenAI での比較分析に失敗しました。',
            };
        }
    }

    async #getToolUsageStatsHandler(_modelSide: ModelSide, args: GetToolUsageStatsArgs): Promise<GetToolUsageStatsResult> {
        const conversationId = (args?.conversation_id ?? '').trim();
        if (!conversationId) {
            return {
                success: false,
                conversation_id: '',
                stats: null,
                error: 'conversation_id を指定してください。',
            };
        }

        const stats = await loadToolUsageStatsFromDirs(this.#logDir, this.#toolStatsDir, conversationId);
        if (!stats) {
            const logExists = await fs.promises.access(`${this.#logDir}/${conversationId}${LOG_FILE_SUFFIX}`)
                .then(() => true)
                .catch(() => false);
            return {
                success: false,
                conversation_id: conversationId,
                stats: null,
                error: logExists
                    ? 'ツール利用統計を取得できませんでした。'
                    : '指定したセッションIDのログが見つかりません。',
            };
        }

        const hasUsage = Object.keys(stats).some(
            actor => stats[actor] && Object.keys(stats[actor]!).length > 0
        );

        return {
            success: true,
            conversation_id: conversationId,
            stats,
            error: hasUsage ? undefined : '記録されたツール利用はありません。',
        };
    }
}

if (IS_MAIN) {
    await PhilosophyDialog.execute({});
}
