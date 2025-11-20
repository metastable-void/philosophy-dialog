
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
    const msgs: RawMessageOpenAi[] = messages.map(msg => {
        if (msg.name == 'anthropic') {
            return {role: 'user', content: msg.content};
        } else {
            return {role: 'assistant', content: msg.content};
        }
    });
    try {
        const count = openaiTokenCounter.chat(msgs, 'cl100k_base');
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
        });
        const output = response.output_text;
        if (!output || 'string' != typeof output) {
            throw new Error('OpenAI didn\'t output text');
        }
        messages.push({
            name: 'openai',
            content: output,
        });
    } catch (e) {
        console.error(e);
        err('openai');
    }
};

const anthropicTurn = async () => {
    const msgs: RawMessage[] = messages.map(msg => {
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
        });
        const tokens = msg.usage.input_tokens + msg.usage.output_tokens;
        if (tokens > CLAUDE_HAIKU_4_5_MAX * 0.8) {
            hushFinish = true;
        }
        const output = msg.content.pop()!;
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

    if (finishTurnCount >= 2) {
        finish();
        break;
    }

    await sleep(SLEEP_BY_STEP);

    if (hushFinish) {
        finishTurnCount += 1;
    }
    await anthropicTurn();
    log('Claude Haiku 4.5', messages[messages.length - 1]!.content);

    if (finishTurnCount >= 2) {
        finish();
        break;
    }

    await sleep(SLEEP_BY_STEP);
}