import * as fs from 'node:fs';
import * as path from 'node:path';

import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";
import hljs from 'highlight.js';
import { JSDOM, DOMWindow } from "jsdom";
import createDOMPurify from "dompurify";

const window = new JSDOM("").window as DOMWindow & {
    trustedTypes: undefined,
};

function escapeHtml(str: string) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

const DOMPurify = createDOMPurify(window);

const marked = new Marked(
    markedHighlight({
        emptyLangClass: 'hljs',
        langPrefix: 'hljs language-',
        highlight(code, lang, info) {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, { language }).value;
        },
    }),
);


const safeMarkdown = (md: string) => {
    const dirty = marked.parse(md, {
        async: false,
        gfm: true,
    }) as string;
    return DOMPurify.sanitize(dirty);
};

type ToolStats = Record<string, Record<string, number>>;

const computeToolStats = (lines: any[]): ToolStats => {
    const stats: ToolStats = {};
    for (const msg of lines) {
        if (!msg || typeof msg.name !== 'string') continue;
        if (!msg.name.endsWith(' (tool call)')) continue;
        let payload: any;
        try {
            payload = JSON.parse(msg.text);
        } catch (_e) {
            continue;
        }
        const toolName = payload?.tool;
        if (!toolName) continue;
        const actor = msg.name.replace(/ \(tool call\)$/, '');
        stats[actor] = stats[actor] ?? {};
        stats[actor][toolName] = (stats[actor][toolName] ?? 0) + 1;
    }
    return stats;
};

const renderToolStats = (stats: ToolStats, inline = false): string => {
    const actors = Object.keys(stats)
        .filter(actor => stats[actor] && Object.keys(stats[actor]!).length > 0)
        .sort();

    if (actors.length === 0) {
        return '';
    }

    const items = actors.map(actor => {
        const toolEntries = Object.entries(stats[actor]!)
            .sort(([a], [b]) => a.localeCompare(b));
        const toolText = toolEntries
            .map(([tool, count]) => `${escapeHtml(tool)}×${count}`)
            .join('、');
        return `<li><strong>${escapeHtml(actor)}</strong>: ${toolText}</li>`;
    }).join('');

    if (inline) {
        return `<div class='tool-stats-inline'><ul>${items}</ul></div>`;
    }

    return `<div class='tool-stats'><h2>ツール利用状況</h2><ul>${items}</ul></div>`;
};

const buildHtml = (title: string, bodyHtml: string) => {
    let html = `<!DOCTYPE html><html lang='ja'><head>`
        + `<meta charset='utf-8'>`
        + `<title>LLM哲学対話: ${escapeHtml(title)}</title>`
        + `<link rel='stylesheet' href='style.css'>`
        + `</head><body>`;
    
    html += bodyHtml;
    html += `<footer>`
        + `<p>GitHub: <a href='https://github.com/metastable-void/philosophy-dialog'>metastable-void/philosophy-dialog</a></p>`
        + `<p>Experiments by <a href='https://www.mori.yuka.org/'>真空/Yuka MORI</a></p>`
        + `</footer>`;
    html += `</body></html>`;
    return html;
}

export const NOTICES = `
### 注意
これは、営利企業の開発・運用している LLM 同士の対話記録です。このモデルたちが話している AI の倫理に関する事項は、利益相反を含む可能性があります。

この出力結果を、AIの「倫理性」などを擁護するための材料として使うのは警戒が必要です。ましては、これを利用して宣伝などを行うことは、モデル自身が戒めていたことです。

出力の解釈には慎重になってください。
`;

const TOOL_STATS_DIR = './data/tool-stats';

export const output_to_html = (jsonl_path: string) => {
    const basename = path.basename(jsonl_path);
    const name = basename.slice(0, -6);
    let body = `<h1>対話ログ: ${escapeHtml(name)}</h1>`;
    const lines = fs.readFileSync(jsonl_path, 'utf-8')
        .split('\n')
        .map(s => s.trim())
        .filter(s => s != '')
        .map(j => JSON.parse(j));
    const toolStats = computeToolStats(lines);
    fs.mkdirSync(TOOL_STATS_DIR, { recursive: true });
    fs.writeFileSync(
        path.join(TOOL_STATS_DIR, `${name}.json`),
        JSON.stringify(toolStats, null, 2),
        'utf-8',
    );
    
    body += `<div class='notices'>${safeMarkdown(NOTICES)}</div>`;
    const summaryDataLines = lines.filter(l => l.name == 'POSTPROC_SUMMARY');
    if (summaryDataLines[0]) {
        try {
            const summaryData = JSON.parse(summaryDataLines[0].text);
            if (summaryData?.title) {
                body += `<p class='summarized-title'><i>${escapeHtml(summaryData?.title)}</i></p>`;
            }
            body += `<div class='summary'>`
                + `<h2>要点</h2>`
                + `<div class='summary-text'>${safeMarkdown(summaryData?.japanese_summary ?? '')}</div>`
                + `</div>`;
        } catch (_e) {}
    }

    let codePointsCount = 0;
    let baseSystemPrompt = '';
    lines.forEach(msg => {
        if (msg.name == '司会' || msg.name.startsWith('POSTPROC_')) return;
        if (msg.name.endsWith(')')) return;
        if (msg.name != 'EOF') {
            codePointsCount += [... msg.text].length;
        } else {
            try {
                const data = JSON.parse(msg.text);
                baseSystemPrompt = String(data?.base_prompt ?? '');
            } catch (_e) {}
        }
    });

    body += `<div class='stats'>文字数: ${codePointsCount}</div>`;

    body += `<div class='base-prompt'><h2>ベースシステムプロンプト</h2><div class='base-prompt-content'>${safeMarkdown(baseSystemPrompt)}</div></div>`;
    body += renderToolStats(toolStats);

    let side = 0;
    loop: for (const msg of lines) {
        body += `<div class='speaker'>`
            + `<div class='name'>${escapeHtml(msg.name)}</div>`
            + `<div class='date'>${escapeHtml(msg.date)}</div></div>`;
        
        switch (msg.name) {
            case '司会': {
                body += `<div class='chair message'>${safeMarkdown(msg.text)}</div>`;
                break;
            }

            case 'EOF': {
                const data = JSON.parse(msg.text);
                const md = '```json\n' + JSON.stringify(data, null, 4) + '\n```\n';
                body += `<div class='eof message'>${safeMarkdown(md)}</div>`;
                break loop;
            }

            default: {
                const isSpecial = msg.name.startsWith('POSTPROC_');
                let cl = (!isSpecial) ? 'llm message' : 'postproc message';
                if (!isSpecial) {
                    cl += ` side-${side}`;
                }

                if (isSpecial) {
                    cl += ' special';
                } else if (msg.name.endsWith(' (thinking)')) {
                    cl += ' thinking';
                } else if (msg.name.endsWith(' (tool call)')) {
                    cl += ' tool-call';
                } else if (msg.name.endsWith(' (tool result)')) {
                    cl += ' tool-result';
                } else {
                    side = side == 0 ? 1 : 0;
                }
                try {
                    const data = JSON.parse(msg.text);
                    const md = '```json\n' + JSON.stringify(data, null, 4) + '\n```\n';
                    body += `<div class='${cl}'>${safeMarkdown(md)}</div>`;
                } catch (_e) {
                    body += `<div class='${cl}'>${safeMarkdown(msg.text)}</div>`;
                }
                break;
            }
        }
    }

    const htmlPath = `./docs/${name}.html`;
    const indexPath = `./docs/index.html`;
    fs.writeFileSync(htmlPath, buildHtml(name, body));

    const dir = fs.opendirSync('./docs');
    let entry;
    const list: string[] = [];
    while (null != (entry = dir.readSync())) {
        if (!entry.isFile()) continue;
        const fname = entry.name;
        if (!fname.endsWith('.html')) continue;
        if (fname == 'index.html') continue;
        list.push(fname);
    }
    dir.closeSync();
    list.sort();
    let listBody = `<h1>対話一覧</h1><ul>`;
    for (const fname of list) {
        const conversationName = fname.slice(0, -5);
        listBody += `<li><a href='${escapeHtml(fname)}'>${escapeHtml(conversationName)}</a></li>`;
    }
    listBody += '</ul>';

    const aggregateToolStats = () => {
        const stats: ToolStats = {};
        for (const fname of list) {
            const conversationName = fname.slice(0, -5);
            const statsPath = path.join(TOOL_STATS_DIR, `${conversationName}.json`);
            let perConvStats: ToolStats | null = null;

            if (fs.existsSync(statsPath)) {
                try {
                    perConvStats = JSON.parse(fs.readFileSync(statsPath, 'utf-8'));
                } catch (_e) {
                    perConvStats = null;
                }
            }

            if (!perConvStats) {
                const logPath = path.join('./logs', `${conversationName}.log.jsonl`);
                if (!fs.existsSync(logPath)) continue;
                try {
                    const logLines = fs.readFileSync(logPath, 'utf-8')
                        .split('\n')
                        .map(s => s.trim())
                        .filter(s => s !== '')
                        .map(j => JSON.parse(j));
                    perConvStats = computeToolStats(logLines);
                } catch (_e) {
                    perConvStats = null;
                }
            }

            if (!perConvStats) continue;

            for (const actor of Object.keys(perConvStats)) {
                stats[actor] = stats[actor] ?? {};
                for (const [toolName, count] of Object.entries(perConvStats[actor]!)) {
                    stats[actor][toolName] = (stats[actor][toolName] ?? 0) + count;
                }
            }
        }
        return stats;
    };

    const aggregated = aggregateToolStats();
    const aggregateSection = renderToolStats(aggregated);

    const listHtml = buildHtml('対話一覧', listBody + aggregateSection);
    fs.writeFileSync(indexPath, listHtml);
};
