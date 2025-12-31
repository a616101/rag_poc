import { marked } from 'marked';
import DOMPurify from 'dompurify';

// 設定 marked 選項
marked.setOptions({
  breaks: true,        // 換行轉 <br>
  gfm: true,           // GitHub Flavored Markdown
  headerIds: false,    // 不產生 header ID（減少體積）
  mangle: false        // 不混淆 email
});

/**
 * 修正 CJK（中日韓）字元的 markdown 粗體/斜體語法
 * marked.js 遵循 CommonMark 規範，對中文字元緊鄰 ** 或 * 的情況解析有問題
 * 此函數在 ** 和中文字元之間插入零寬空格以修正此問題
 *
 * @param {string} content
 * @returns {string}
 */
function normalizeMarkdownForCJK(content) {
  if (!content) return content;

  // CJK 字元範圍
  const cjkRange = '\\u4e00-\\u9fff\\u3400-\\u4dbf\\uf900-\\ufaff\\u3000-\\u303f\\uff00-\\uffef';

  // 處理粗體 **text**
  let result = content
    // **後接CJK：**中 → ** 中（加零寬空格）
    .replace(new RegExp(`(\\*\\*)([${cjkRange}])`, 'g'), '$1\u200B$2')
    // CJK後接**：中** → 中 **（加零寬空格）
    .replace(new RegExp(`([${cjkRange}])(\\*\\*)`, 'g'), '$1\u200B$2');

  // 處理斜體 *text*（單星號，但要避免影響粗體）
  result = result
    .replace(new RegExp(`(?<!\\*)(\\*)(?!\\*)([${cjkRange}])`, 'g'), '$1\u200B$2')
    .replace(new RegExp(`([${cjkRange}])(?<!\\*)(\\*)(?!\\*)`, 'g'), '$1\u200B$2');

  return result;
}

/**
 * 將 Markdown 轉換為安全的 HTML
 *
 * @param {string} content - Markdown 內容
 * @returns {string} 安全的 HTML
 */
export function renderMarkdown(content) {
  if (!content) return '';

  // 1. 修正 CJK 語法問題
  const normalized = normalizeMarkdownForCJK(content);

  // 2. 轉換為 HTML
  const html = marked.parse(normalized);

  // 3. 使用 DOMPurify 清理（防止 XSS）
  const clean = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'b', 'em', 'i', 'u',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li',
      'a', 'code', 'pre', 'blockquote',
      'table', 'thead', 'tbody', 'tr', 'th', 'td',
      'img', 'hr'
    ],
    ALLOWED_ATTR: ['href', 'target', 'rel', 'src', 'alt', 'title', 'class', 'data-zoomable', 'style']
  });

  // 4. 後處理：連結另開新視窗、圖片加上可放大標記
  const processed = postProcessHtml(clean);

  return processed;
}

/**
 * HTML 後處理
 * - 連結加上 target="_blank" 和 rel="noopener noreferrer"
 * - 圖片加上 data-zoomable 屬性
 *
 * @param {string} html
 * @returns {string}
 */
function postProcessHtml(html) {
  // 使用 DOMParser 處理
  const parser = new DOMParser();
  const doc = parser.parseFromString(`<div>${html}</div>`, 'text/html');
  const container = doc.body.firstChild;

  // 處理所有連結
  container.querySelectorAll('a').forEach(link => {
    // 外部連結另開新視窗
    const href = link.getAttribute('href');
    if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
    }
  });

  // 處理所有圖片 - 加上可放大標記
  container.querySelectorAll('img').forEach(img => {
    img.setAttribute('data-zoomable', 'true');
    img.style.cursor = 'zoom-in';
  });

  return container.innerHTML;
}

/**
 * 純文字預覽（用於通知等場景）
 *
 * @param {string} content - Markdown 內容
 * @param {number} maxLength - 最大長度
 * @returns {string}
 */
export function getPlainTextPreview(content, maxLength = 100) {
  if (!content) return '';

  // 移除 Markdown 語法
  const plain = content
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/#{1,6}\s+/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\n+/g, ' ')
    .trim();

  if (plain.length <= maxLength) return plain;
  return plain.slice(0, maxLength) + '...';
}
