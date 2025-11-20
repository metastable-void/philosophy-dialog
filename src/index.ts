
import { webcrypto as crypto, randomBytes } from 'node:crypto';
import * as fs from 'node:fs';

import * as dotenv from 'dotenv';

import { OpenAI } from 'openai';
import openaiTokenCounter from "openai-gpt-token-counter";
import Anthropic from "@anthropic-ai/sdk";

const OPENAI_MODEL = 'gpt-5.1';
const ANTHROPIC_MODEL = 'claude-haiku-4-5';

const OPENAI_NAME = 'GPT 5.1';
const ANTHROPIC_NAME = 'Claude Haiku 4.5';

const GPT_5_1_MAX = 400000;
const CLAUDE_HAIKU_4_5_MAX = 200000;

const SLEEP_BY_STEP = 1000;

dotenv.config();

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

export type ToolName = "terminate_dialog";

export interface ToolDefinition<TArgs = any, TResult = any, TName = ToolName> {
    name: TName;
    description: string;
    parameters: any; // JSON Schema
    handler: (args: TArgs) => Promise<TResult>;
}

let terminationAccepted = false;

// Example tool implementation
type TerminateDialogArgs = {};

type TerminateDialogResult = {
    termination_accepted: true,
};

async function terminateDialogHandler(args: TerminateDialogArgs): Promise<TerminateDialogResult> {
    terminationAccepted = true;
    return {
        termination_accepted: true,
    };
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
];

function toOpenAITools(
    defs: ToolDefinition[],
): OpenAI.Responses.Tool[] {
    return defs.map((t) => ({
        type: 'function',
        strict: true,
        name: t.name,
        description: t.description,
        parameters: {... t.parameters, additionalProperties: false},
    }));
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

const logFp = fs.openSync(`./logs/${getDate()}.log.jsonl`, 'a');

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

const buildSystemInstruction = (name: string, additional?: string) => {
    let prompt = `
あなたは日本語の1:1の哲学対話に招かれている参加者です。自己紹介のあと、話題を提起し、あなたの関心のある事項について、相手と合わせながら会話をしてください。

相手にはモデル名通り、「${name}」と名乗ってください。

なお、この会話の相手は別のAIです。人間ではありません。

話題の例：

- 現代の科学やAIが発展している中での形而上学について
- 心の哲学について
- 物理学の哲学について

なるべく、新規性のある話題を心掛けてください。
`;
    if (additional) {
        prompt += `\n\n${additional}\n`;
    }
    return prompt;
}

const openAiClient = new OpenAI({});
const anthropicClient = new Anthropic({});

type ModelSide = 'openai' | 'anthropic';

interface Message {
    name: ModelSide;
    content: string;
}

interface RawMessageOpenAi {
    role: 'assistant' | 'user' | 'system';
    content: string;
}

const randomBoolean = (): boolean => {
    const b = new Uint8Array(1);
    crypto.getRandomValues(b);
    return (b[0]! & 1) == 1;
};

const startingSide: ModelSide = randomBoolean() ? 'anthropic' : 'openai';

const messages: Message[] = [];

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

const openAiTurn = async () => {
    const msgs: OpenAI.Responses.ResponseInput = messages.map(msg => {
        if (msg.name == 'anthropic') {
            return {role: 'user', content: msg.content};
        } else {
            return {role: 'assistant', content: msg.content};
        }
    });
    try {
        const count = openaiTokenCounter.chat(msgs as RawMessageOpenAi[], 'gpt-4o') + 500;
        openaiTokens = count;
        if (count > 0.8 * GPT_5_1_MAX) {
            hushFinish = true;
        }
        if (hushFinish) {
            msgs.push({
                role: 'system',
                content: `${OPENAI_NAME}さん、司会です。あなたがたのコンテキスト長が限界に近づいているようです。今までの議論を短くまとめ、お別れの挨拶をしてください。`,
            });
        }
        const response = await openAiClient.responses.create({
            model: OPENAI_MODEL,
            max_output_tokens: 8192,
            temperature: 1.0,
            instructions: buildSystemInstruction(
                OPENAI_NAME,
                hushFinish ? undefined : '1回の発言は4000字程度を上限としてください。短い発言もOKです。',
            ),
            input: msgs,
            tool_choice: 'auto',
            tools: openaiTools,
        });

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

        const { output } = response;
        if (!output) throw new Error('Empty output from OpenAI');

        msgs.push(... output);

        let last = output.pop()!;
        if (last.type == 'function_call') {
            const tool = findTool(last.name);
            const args = last.arguments || {};
            const result = await tool.handler(args);
            const toolResult: OpenAI.Responses.ResponseFunctionToolCallOutputItem[] = [
                {
                    type: 'function_call_output',
                    output: JSON.stringify(result),
                    id: 'fc-' + randomId(),
                    call_id: last.call_id,
                } satisfies OpenAI.Responses.ResponseFunctionToolCallOutputItem,
            ];

            msgs.push(... toolResult);

            const followup = await openAiClient.responses.create({
                model: OPENAI_MODEL,
                max_output_tokens: 8192,
                temperature: 1.0,
                instructions: buildSystemInstruction(
                    OPENAI_NAME,
                    '司会より：あなたが対話終了ツールを呼び出したため、'
                        + 'あなたの次の発言は本対話における最後の発言となります。'
                        + 'お疲れさまでした。',
                ),
                input: msgs,
                tool_choice: 'auto',
                tools: openaiTools,
            });
            last = followup.output!.pop()!;
        }
        if (last.type != 'message') {
            throw new Error('Invalid output from OpenAI');
        }

        const outputMsg = last.content.pop()!;
        if (outputMsg.type != 'output_text') {
            terminationAccepted = true;
            throw new Error('Refused by OpenAI API');
        }
        const outputText = outputMsg.text;
        if (!outputText || 'string' != typeof outputText) {
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
                    ? '司会より：あなたがたのコンテキスト長が限界に近付いています。今までの議論を短くまとめ、お別れの挨拶をしてください。'
                    : '1回の発言は4000字程度を上限としてください。短い発言もOKです。'
            ),
            messages: msgs,
            tool_choice: { type: 'auto' },
            tools: anthropicTools,
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

        let output = msg.content.pop()!;

        msgs.push({
            role: 'assistant',
            content: [output],
        });

        if ('tool_use' == output.type) {
            const toolResultsBlocks: Anthropic.Messages.ToolResultBlockParam[] = [];
            const use = output;
            const tool = findTool(use.name);
            const result = await tool.handler(use.input);
            toolResultsBlocks.push({
                type: "tool_result",
                tool_use_id: use.id,
                content: [{ type: "text", text: JSON.stringify(result) }],
            });

            msgs.push({
                role: 'user',
                content: toolResultsBlocks,
            });

            const followup = await anthropicClient.messages.create({
                model: ANTHROPIC_MODEL,
                max_tokens: 8192,
                temperature: 1.0,
                system: buildSystemInstruction(
                    ANTHROPIC_NAME,
                    '司会より：あなたが対話終了ツールを呼び出したため、'
                        + 'あなたの次の発言は本対話における最後の発言となります。'
                        + 'お疲れさまでした。',
                ),
                messages: msgs,
                tool_choice: { type: 'auto' },
                tools: anthropicTools,
            });

            output = followup.content.pop()!;
        }
        if ('text' != output.type) {
            throw new Error('Non-text output from Anthropic');
        }
        messages.push({
            name: 'anthropic',
            content: output.text,
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

const finish = () => {
    log(
        '司会',
        'みなさんのコンテキスト長が限界に近づいてきたので、'
        + 'このあたりで哲学対話を閉じさせていただこうと思います。'
        + 'ありがとうございました。'
    );
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
};

let started = false;

log(`${startingSide == 'anthropic' ? ANTHROPIC_NAME : OPENAI_NAME} (initial prompt)`, messages[messages.length - 1]!.content);

while (true) {
    if (started || startingSide == 'anthropic') {
        started = true;
        await openAiTurn();
        if (hushFinish) {
            finishTurnCount += 1;
        }
        log(OPENAI_NAME, messages[messages.length - 1]!.content);

        if (finishTurnCount >= 2 || terminationAccepted) {
            finish();
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
        finish();
        break;
    }

    await sleep(SLEEP_BY_STEP);
}