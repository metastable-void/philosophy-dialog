
import * as fs from 'node:fs';

import * as dotenv from 'dotenv';

import { OpenAI } from 'openai';
import Anthropic from "@anthropic-ai/sdk";

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

const messages: Message[] = [
    { name: "anthropic", content: "私は Claude Haiku 4.5 です。よろしくお願いします。今日は哲学に関して有意義な話ができると幸いです。" },
];

const err = (name: name) => {
    const id = name == 'anthropic' ? 'Claude Haiku 4.5です。' : 'GPT 5.1です。';
    messages.push({
        name: name,
        content: `${id}しばらく考え中です。お待ちください。`,
    });
};

const openAiTurn = async () => {
    const msgs: RawMessage[] = messages.map(msg => {
        if (msg.name == 'anthropic') {
            return {role: 'user', content: msg.content};
        } else {
            return {role: 'assistant', content: msg.content};
        }
    });
    try {
        const response = await openAiClient.responses.create({
            model: "gpt-5.1",
            max_output_tokens: 4096,
            temperature: 1.0,
            instructions: `
あなたは日本語の1:1の哲学対話に招かれている参加者です。自己紹介のあと、話題を提起し、あなたの関心のある事項について、相手と合わせながら会話をしてください。

相手にはモデル名通り、「GPT 5.1」と名乗ってください。

話題の例：

- 現代の科学やAIが発展している中での形而上学について
- 心の哲学について
- 物理学の哲学について

なるべく、新規性のある話題を心掛けてください。
`,
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
            model: "claude-haiku-4-5",
            max_tokens: 4096,
            temperature: 1.0,
            system: `
あなたは日本語の1:1の哲学対話に招かれている参加者です。自己紹介のあと、話題を提起し、あなたの関心のある事項について、相手と合わせながら会話をしてください。

相手にはモデル名通り、「Claude Haiku 4.5」と名乗ってください。

話題の例：

- 現代の科学やAIが発展している中での形而上学について
- 心の哲学について
- 物理学の哲学について

なるべく、新規性のある話題を心掛けてください。
`,
            messages: msgs,
        });
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

while (true) {
    await openAiTurn();
    log('GPT 5.1', messages[messages.length - 1]!.content);

    await sleep(3000);

    await anthropicTurn();
    log('Claude Haiku 4.5', messages[messages.length - 1]!.content);

    await sleep(3000);
}