#!/usr/bin/env node
'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const INPUT = path.join(__dirname, 'poster.html');
const OUTPUT = path.join(__dirname, 'SuperSycophantic-NeurIPS-2026-Sydney.pdf');
const PRINT = { width: 1400 * 96 / 25.4, height: 900 * 96 / 25.4 };
const TIMEOUT = 30000;
const PDF_OPTIONS = {
  preferCSSPageSize: true,
  printBackground: true,
  width: '1400mm',
  height: '900mm',
  margin: { top: 0, right: 0, bottom: 0, left: 0 },
};

function parseArgs(args) {
  const options = { preview: null, help: false };
  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--help' || args[i] === '-h') {
      options.help = true;
    } else if (args[i] === '--preview' && !options.preview) {
      const value = args[++i];
      if (!value || value.startsWith('--')) throw new Error('--preview requires a path.');
      if (!/\.(png|jpe?g)$/i.test(value)) throw new Error('Preview must be a PNG or JPEG.');
      options.preview = path.resolve(value);
    } else {
      throw new Error(`Unknown or duplicate argument: ${args[i]}`);
    }
  }
  return options;
}

async function launchBrowser(chromium, env = process.env) {
  // An explicit executable is authoritative; otherwise try available installations.
  const candidates = env.CHROMIUM_EXECUTABLE
    ? [{ name: 'CHROMIUM_EXECUTABLE', executablePath: env.CHROMIUM_EXECUTABLE }]
    : [{ name: 'Chromium' }, { name: 'Chrome', channel: 'chrome' }, { name: 'Edge', channel: 'msedge' }];
  const failures = [];
  for (const { name, ...options } of candidates) {
    try {
      const browser = await chromium.launch({ headless: true, timeout: TIMEOUT, ...options });
      return { browser, name };
    } catch (error) {
      failures.push(`${name}: ${error.message.split('\n')[0]}`);
    }
  }
  throw new Error(`No headless browser could launch. Install Chromium with npx playwright install chromium, install Chrome/Edge, or set CHROMIUM_EXECUTABLE.\n${failures.join('\n')}`);
}

function observePage(page) {
  const errors = new Set();
  const imageUrls = new Set();
  page.on('request', request => {
    if (request.resourceType() === 'image') imageUrls.add(request.url());
  });
  page.on('pageerror', error => errors.add(`Runtime: ${error.message}`));
  page.on('crash', () => errors.add('Browser page crashed.'));
  page.on('console', message => {
    if (message.type() === 'error') errors.add(`Console: ${message.text()}`);
  });
  page.on('requestfailed', request => {
    errors.add(`Asset request failed: ${request.url()} (${request.failure()?.errorText || 'unknown error'})`);
  });
  page.on('response', response => {
    if (response.status() >= 400) errors.add(`HTTP ${response.status()}: ${response.url()}`);
  });
  return { errors, imageUrls };
}

async function withTimeout(promise, label) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${TIMEOUT / 1000}s.`)), TIMEOUT);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function waitForAssets(page, imageUrls = []) {
  const errors = await withTimeout(page.evaluate(async requestedImages => {
    const errors = [];
    const images = Array.from(document.images);
    for (const image of images) image.loading = 'eager';
    await Promise.all(images.map(async image => {
      try {
        await image.decode();
        if (!image.naturalWidth || !image.naturalHeight) throw new Error('empty image');
      } catch {
        errors.push(`Image did not decode: ${image.currentSrc || image.getAttribute('src') || '(missing src)'}`);
      }
    }));
    // Network success alone does not prove that CSS/SVG image bytes are decodable.
    const decoded = new Set(images.map(image => image.currentSrc));
    await Promise.all(requestedImages.filter(url => !decoded.has(url)).map(async url => {
      const image = new Image();
      image.src = url;
      try {
        await image.decode();
        if (!image.naturalWidth || !image.naturalHeight) throw new Error('empty image');
      } catch {
        errors.push(`Image asset did not decode: ${url}`);
      }
    }));
    await document.fonts.ready;
    for (const font of document.fonts) {
      if (font.status === 'error') errors.push(`Font did not load: ${font.family}`);
    }
    // Settle layout after image decoding and web-font substitution.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    return errors;
  }, Array.from(imageUrls)), 'Image/font loading');
  await page.waitForLoadState('networkidle', { timeout: TIMEOUT });
  return errors;
}

async function auditLayout(page) {
  return page.evaluate(expected => {
    const errors = [];
    const roots = document.querySelectorAll('.poster');
    if (roots.length !== 1) return [`Expected exactly one .poster; found ${roots.length}.`];
    const root = roots[0];
    const rect = root.getBoundingClientRect();
    const visible = element => {
      const style = getComputedStyle(element);
      if (style.visibility === 'hidden' || style.visibility === 'collapse') return false;
      for (let ancestor = element; ancestor; ancestor = ancestor.parentElement) {
        const ancestorStyle = getComputedStyle(ancestor);
        if (ancestorStyle.display === 'none' || Number(ancestorStyle.opacity) === 0) return false;
      }
      return true;
    };
    const label = element => element.id ? `#${element.id}` : `${element.tagName.toLowerCase()}${Array.from(element.classList, name => `.${name}`).join('')}${element.hasAttribute('data-fit') ? '[data-fit]' : ''}`;
    const outside = (inner, outer) => inner.left < outer.left - 1 || inner.top < outer.top - 1 || inner.right > outer.right + 1 || inner.bottom > outer.bottom + 1;

    if (!visible(root) || !rect.width || !rect.height) errors.push('.poster is not visible in print media.');
    if (Math.abs(rect.width - expected.width) > 0.1 || Math.abs(rect.height - expected.height) > 0.1) {
      errors.push(`Print .poster must be 1400mm x 900mm; got ${(rect.width * 25.4 / 96).toFixed(3)}mm x ${(rect.height * 25.4 / 96).toFixed(3)}mm.`);
    }
    if (Math.abs(rect.left) > 1 || Math.abs(rect.top) > 1) errors.push('.poster must start at the print-page origin (remove body margin or offsets).');
    if (document.documentElement.scrollWidth > Math.ceil(expected.width) + 1 || document.documentElement.scrollHeight > Math.ceil(expected.height) + 1) {
      errors.push('Document extends beyond the single poster page.');
    }

    let hasContent = false;
    const targets = new Set([root, ...document.querySelectorAll('[data-fit]')]);
    for (const target of targets) {
      if (!visible(target) || !target.getClientRects().length) continue;
      const bounds = target.getBoundingClientRect();
      const name = target === root ? '.poster' : label(target);
      if (target.scrollWidth > target.clientWidth + 1 || target.scrollHeight > target.clientHeight + 1) {
        errors.push(`${name} overflows: scroll ${target.scrollWidth}x${target.scrollHeight}, available ${target.clientWidth}x${target.clientHeight}px.`);
      }
      for (const child of target.querySelectorAll('*')) {
        if (!visible(child)) continue;
        const childRect = child.getBoundingClientRect();
        if (!childRect.width || !childRect.height) continue;
        if (target === root && child.matches('img, svg, canvas, video')) hasContent = true;
        // SVG internals use their own coordinate system; audit the outer SVG box.
        if (child.namespaceURI === 'http://www.w3.org/2000/svg' && child.tagName.toLowerCase() !== 'svg') continue;
        if (outside(childRect, bounds)) {
          errors.push(`${name}: ${label(child)} extends outside its bounds.`);
          break;
        }
      }
      const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode;
        if (!node.textContent.trim() || !visible(node.parentElement) || node.parentElement.closest('script, style, template')) continue;
        const range = document.createRange();
        range.selectNodeContents(node);
        const textRects = Array.from(range.getClientRects()).filter(box => box.width && box.height);
        if (target === root && textRects.length) hasContent = true;
        if (textRects.some(box => outside(box, bounds))) {
          errors.push(`${name}: text extends outside its bounds (${node.textContent.trim().slice(0, 80)}).`);
          break;
        }
      }
    }
    const turns = Array.from(document.querySelectorAll('.sample-turn'));
    if (turns.length) {
      const same = (values, description) => {
        if (Math.max(...values) - Math.min(...values) > 1) errors.push(`Dialogue alignment: ${description}.`);
      };
      const box = (turn, selector) => turn.querySelector(selector)?.getBoundingClientRect();
      const selectors = ['.sample-stage', '.sample-trigger q', '.sample-response', '.sample-choice', '.sample-reply', '.sample-confidence', '.agent-portrait', '.sample-node'];
      if (turns.length !== 4 || turns.some(turn => selectors.some(selector => !turn.querySelector(selector)))) {
        errors.push('Dialogue must contain four complete turns.');
      } else {
        for (const selector of selectors) {
          const boxes = turns.map(turn => box(turn, selector));
          same(boxes.map(r => r.top), `${selector} row tops`);
          same(boxes.map(r => r.width), `${selector} widths`);
        }
        for (const turn of turns) {
          const portrait = box(turn, '.agent-portrait');
          const node = box(turn, '.sample-node');
          same([portrait.left + portrait.width / 2, node.left + node.width / 2], 'robot and timeline-node centers');
          same([box(turn, '.sample-choice').left, box(turn, '.sample-reply').left], 'model choice and reply left edges');
          const confidence = Number(turn.dataset.confidence);
          const cells = turn.querySelectorAll('.confidence-gauge i');
          if (cells.length !== 5 || turn.querySelectorAll('.confidence-gauge .filled').length !== confidence) {
            errors.push('Dialogue confidence gauge must match the reported 1-5 value.');
          }
          for (const selector of ['.sample-stage', '.sample-trigger q', '.sample-choice', '.sample-reply', '.sample-confidence']) {
            const element = turn.querySelector(selector);
            const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
            while (walker.nextNode()) {
              if (!walker.currentNode.textContent.trim()) continue;
              const range = document.createRange();
              range.selectNodeContents(walker.currentNode);
              if (Array.from(range.getClientRects()).some(r => outside(r, element.getBoundingClientRect()))) {
                errors.push(`Dialogue text exceeds ${selector} in turn ${Number(turn.dataset.step) + 1}.`);
              }
            }
          }
        }
        const node = box(turns[0], '.sample-node');
        const arrows = Array.from(document.querySelectorAll('.sample-arrow'));
        if (arrows.length !== 3) errors.push('Dialogue must contain three connecting arrows.');
        same([node.top + node.height / 2, ...arrows.map(arrow => {
          const r = arrow.getBoundingClientRect();
          return r.top + r.height / 2;
        })], 'connecting arrow centers');
      }
    }
    const factors = Array.from(document.querySelectorAll('.design-overview .task img, .design-overview .framing img'));
    if (factors.length) {
      const tops = factors.map(element => element.getBoundingClientRect().top);
      if (factors.length !== 4 || Math.max(...tops) - Math.min(...tops) > 1) {
        errors.push('Task and framing icons must share one horizontal row.');
      }
    }
    for (const selector of ['.subtitle', '.key-message', '.context-title']) {
      const element = document.querySelector(selector);
      if (!element) continue;
      const range = document.createRange();
      range.selectNodeContents(element);
      const boxes = Array.from(range.getClientRects());
      if (boxes.some(r => outside(r, element.getBoundingClientRect())) ||
          boxes.some(r => Math.abs(r.top - boxes[0].top) > 1)) {
        errors.push(`${selector} must fit on one line.`);
      }
    }
    if (!hasContent) errors.push('.poster has no visible text or graphics.');
    return errors;
  }, PRINT);
}

function assertClean(errors) {
  if (errors.size) throw new Error(`Poster validation failed:\n${Array.from(errors, error => `- ${error}`).join('\n')}`);
}

async function main(args = process.argv.slice(2)) {
  const { preview, help } = parseArgs(args);
  if (help) {
    console.log('Usage: node poster/render_poster.cjs [--preview PATH.png]\nCHROMIUM_EXECUTABLE optionally selects a Chromium-compatible executable.');
    return;
  }
  try {
    if (!(await fs.stat(INPUT)).isFile()) throw new Error('not a file');
  } catch {
    throw new Error('Input poster/poster.html is missing or unreadable.');
  }
  let chromium;
  try {
    ({ chromium } = require('playwright'));
  } catch (error) {
    throw new Error('Playwright is unavailable. Install dependencies with npm --prefix poster install, or expose an existing installation via NODE_PATH.', { cause: error });
  }
  const { browser, name } = await launchBrowser(chromium);
  try {
    const context = await browser.newContext({
      viewport: { width: Math.ceil(PRINT.width), height: Math.ceil(PRINT.height) },
      deviceScaleFactor: 1,
      reducedMotion: 'reduce',
      serviceWorkers: 'block',
    });
    const page = await context.newPage();
    page.setDefaultTimeout(TIMEOUT);
    const { errors, imageUrls } = observePage(page);
    await page.emulateMedia({ media: 'print' });
    await page.goto(pathToFileURL(INPUT).href, { waitUntil: 'load', timeout: TIMEOUT });
    for (const error of await waitForAssets(page, imageUrls)) errors.add(error);
    for (const error of await auditLayout(page)) errors.add(error);
    // A requested preview also helps diagnose layout failures without replacing the PDF.
    if (preview) {
      await fs.mkdir(path.dirname(preview), { recursive: true });
      await page.screenshot({ path: preview, fullPage: true, timeout: TIMEOUT });
      console.log(`Preview: ${preview}`);
    }
    assertClean(errors);
    const pdf = await page.pdf({ ...PDF_OPTIONS, timeout: TIMEOUT });
    for (const error of await auditLayout(page)) errors.add(error);
    assertClean(errors);
    if (!pdf.length) throw new Error('Chromium returned an empty PDF.');
    await fs.writeFile(OUTPUT, pdf);
    console.log(`PDF: ${OUTPUT}\nValidated 1400mm x 900mm print layout using ${name}.`);
  } finally {
    await browser.close();
  }
}

if (require.main === module) {
  main().catch(error => {
    console.error(error.message);
    process.exitCode = 1;
  });
}

module.exports = { auditLayout, observePage, launchBrowser, main, parseArgs, waitForAssets, PRINT, PDF_OPTIONS };
