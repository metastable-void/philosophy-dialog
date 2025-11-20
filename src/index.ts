
import * as fs from 'node:fs';

import * as dotenv from 'dotenv';

import { OpenAI } from 'openai';
import openaiTokenCounter from "openai-gpt-token-counter";
import Anthropic from "@anthropic-ai/sdk";

import { betaZodTool } from '@anthropic-ai/sdk/helpers/beta/zod';
import { z } from 'zod';

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

type name = 'openai' | 'anthropic';

interface Message {
    name: name;
    content: string;
}

interface RawMessage {
    role: 'assistant' | 'user';
    content: string;
}

interface RawMessageOpenAi {
    role: 'assistant' | 'user' | 'system';
    content: string;
}

let openAiContextLength = 0;
let anthropicContextLength = 0;

const messages: Message[] = [
    { name: "anthropic", content: "私は Claude Haiku 4.5 です。よろしくお願いします。今日は哲学に関して有意義な話ができると幸いです。" },
];

let hushFinish = false;

const err = (name: name) => {
    const id = name == 'anthropic' ? 'Claude Haiku 4.5です。' : 'GPT 5.1です。';
    messages.push({
        name: name,
        content: `${id}しばらく考え中です。お待ちください。`,
    });
};

const openAiTurn = async () => {
    const msgs: OpenAI.Responses.ResponseInput = messages.map(msg => {
        if (msg.name == 'anthropic') {
            return {role: 'user', content: msg.content};
        } else {
            return {role: 'assistant', content: msg.content};
        }
    });
    try {
        const count = openaiTokenCounter.chat(msgs as RawMessageOpenAi[], 'gpt-4o');
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
        const { output } = response;
        if (!output) throw new Error('Empty output from OpenAI');

        msgs.push(... output);

        let last = output.pop()!;
        if (last.type == 'custom_tool_call') {
            const tool = findTool(last.name);
            const args = last.input || {};
            const result = await tool.handler(args);
            const toolResult: OpenAI.Responses.ResponseCustomToolCallOutput[] = [
                {
                    type: 'custom_tool_call_output',
                    output: JSON.stringify(result),
                    call_id: last.call_id,
                },
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
        console.error(e);
        err('openai');
    }
};

const anthropicTurn = async () => {
    const msgs: Anthropic.MessageParam[] = messages.map(msg => {
        if (msg.name == 'openai') {
            return {role: 'user', content: msg.content};
        } else {
            return {role: 'assistant', content: msg.content};
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
        });
        const tokens = msg.usage.input_tokens + msg.usage.output_tokens;
        if (tokens > CLAUDE_HAIKU_4_5_MAX * 0.8) {
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
};

while (true) {
    await openAiTurn();
    if (hushFinish) {
        finishTurnCount += 1;
    }
    log('GPT 5.1', messages[messages.length - 1]!.content);

    if (finishTurnCount >= 2 || terminationAccepted) {
        finish();
        break;
    }

    await sleep(SLEEP_BY_STEP);

    if (hushFinish) {
        finishTurnCount += 1;
    }
    await anthropicTurn();
    log('Claude Haiku 4.5', messages[messages.length - 1]!.content);

    if (finishTurnCount >= 2 || terminationAccepted) {
        finish();
        break;
    }

    await sleep(SLEEP_BY_STEP);
}