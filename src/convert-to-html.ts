#!/usr/bin/env node

import * as fs from 'node:fs';
import { output_to_html } from './html.js';

const path = process.argv[2];
if (!path) {
    fs.writeSync(1, `Usage: ${process.argv[0]} ${process.argv[1]} <jsonl_path>\n`);
    process.exit(0);
}

output_to_html(path);
