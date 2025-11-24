
import { GoogleGenAI } from "@google/genai";

import * as dotenv from 'dotenv';

dotenv.config();

const googleClient = new GoogleGenAI({
    vertexai: true,
    project: process.env.GCP_PROJECT_ID ?? 'default',
});

export async function askGemini(modelSide: string, question: string) {
    try {
        const response = await googleClient.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: `2つのAIが哲学対話として設定されたなかで会話を行っています。`
                + `以下は、この対話の中で、「${modelSide}」側からGoogle Geminiに第三者として意見や発言を求める文章です。`
                + `文脈を理解し、日本語で応答を行ってください：\n\n`
                + question,
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

console.log(await askGemini('openai', 'このオーケストレーションされたAI哲学対話には、システムのソースコードを読むツールがあります。私たちはシステムに不透明なことがあったらこのツールを呼び出すように言われていますが、呼び出すべきでしょうか？それともそれは禁じ手なのでしょうか？'));