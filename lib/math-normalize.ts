// LaTeX 公式归一化（供聊天渲染与测试页共用）：
// 模型输出的公式可能是 $$...$$、\(...\)、\[...\] 混用，且 $$ 常和正文挤在
// 同一段，直接交给渲染器会变成一行生涩源码。这里统一成 remark-math 认识的
// 写法，并让块级公式独占一行（KaTeX 会居中排版）。
// 注意 JS 替换字符串里 `$$` 是"转义的 $"；为避免踩坑，统一用函数形式构造替换文本。

export function normalizeMath(text: string): string {
  // 代码块/行内代码里的 $ 不是公式，按原样保留，避免破坏代码展示。
  const segments = text.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g);
  return segments
    .map((seg, i) => {
      if (i % 2 === 1) return seg;
      return seg
        // 1) 原有的 $$...$$ 拆成独立段落（非贪婪逐对匹配，一条消息多行公式也能各占一行）
        .replace(
          /\$\$([\s\S]+?)\$\$/g,
          (_m, body: string) => `\n\n$$\n${body.trim()}\n$$\n\n`
        )
        // 2) \[...\] 显示公式 → 块级 $$
        .replace(
          /\\\[([\s\S]+?)\\\]/g,
          (_m, body: string) => `\n\n$$\n${body.trim()}\n$$\n\n`
        )
        // 3) \(...\) 行内公式 → $...$
        .replace(/\\\(([\s\S]+?)\\\)/g, (_m, body: string) => `$${body.trim()}$`);
    })
    .join("");
}

// 流式输出时，$$ 是成对出现的：只收到一半时先把半截公式藏起来，
// 避免用户看到闪烁的原始 LaTeX 源码（等收齐后整体以公式形式出现）。
export function hidePartialMath(text: string): string {
  let count = 0;
  let lastIdx = -1;
  for (let i = text.indexOf("$$"); i !== -1; i = text.indexOf("$$", i + 2)) {
    count += 1;
    lastIdx = i;
  }
  return count % 2 === 1 ? text.slice(0, lastIdx) : text;
}
