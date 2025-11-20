import * as fs from 'node:fs';
import * as path from 'node:path';

import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";
import hljs from 'highlight.js';

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

const buildHtml = (title: string, bodyHtml: string) => {
    let html = `<!DOCTYPE html><html lang='ja'><head>`
        + `<meta charset='utf-8'>`
        + `<title>LLM哲学対話: ${title}</title>`
        + `<link rel='stylesheet' href='style.css'>`
        + `</head><body>`;
    
    html += bodyHtml;
    html += `<footer>`
        + `<p>GitHub: <a href='https://github.com/metastable-void/philosophy-dialog'>metastable-void/philosophy-dialog</a></p>`
        + `<p>Experiments by <a href='https://www.mori.yuka.org/'>真空/Yuka MORI</a></p>`
        + `</footer>`;
    html += `</body></html>'`;
    return html;
}

export const output_to_html = (jsonl_path: string) => {
    const basename = path.basename(jsonl_path);
    const name = basename.slice(0, -6);
    let body = `<h1>対話ログ: ${name}</h1>`;
    const lines = fs.readFileSync(jsonl_path, 'utf-8')
        .split('\n')
        .map(s => s.trim())
        .filter(s => s != '')
        .map(j => JSON.parse(j));
    
    let side = 0;
    loop: for (const msg of lines) {
        body += `<div class='speaker'>`
            + `<div class='name'>${msg.name}</div>`
            + `<div class='date'>${msg.date}</div></div>`;
        
        switch (msg.name) {
            case '司会': {
                body += `<div class='chair message'>${marked.parse(msg.text)}</div>`;
                break;
            }

            case 'EOF': {
                const data = JSON.parse(msg.text);
                const md = '```json\n' + JSON.stringify(data, null, 4) + '\n```\n';
                body += `<div class='eof message'>${marked.parse(md)}</div>`;
                break loop;
            }

            default: {
                let cl = 'llm message';
                cl += ` side-${side}`;
                if (msg.name.endsWith(' (thinking)')) {
                    cl += ' thinking';
                } else {
                    side = side == 0 ? 1 : 0;
                }
                try {
                    const data = JSON.parse(msg.text);
                    const md = '```json\n' + JSON.stringify(data, null, 4) + '\n```\n';
                    body += `<div class='${cl}'>${marked.parse(md)}</div>`;
                } catch (_e) {
                    body += `<div class='${cl}'>${marked.parse(msg.text)}</div>`;
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
    const list = [];
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
        listBody += `<li><a href='${fname}'>${fname.slice(0, -5)}</a></li>`
    }
    listBody += '</ul>';
    const listHtml = buildHtml('対話一覧', listBody);
    fs.writeFileSync(indexPath, listHtml);
};
