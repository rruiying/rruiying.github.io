# 怎么发博客（网页上就能完成）

每篇文章 = `content/posts/` 里的**一个 Markdown 文件**。推送后 GitHub Actions
会自动生成文章页面并更新博客索引 —— `posts/` 目录和 `posts.js` 都是机器生成的,
**永远不要手改**。

## 发一篇新文章

1. 打开仓库页面 → 进入 `content/posts/` → **Add file → Create new file**
2. 文件名用英文小写加连字符，如 `convex-optimization.md`（它决定文章网址）
3. 按这个格式写：

```markdown
---
title: Convex Optimization — lecture notes
date: 2026-09-15
category: Course notes
tags: [optimization, math, TUM]
summary: 一两句摘要，显示在博客列表里，参与搜索。
---
正文从这里开始，就是普通 Markdown。

## 小节标题

支持 **加粗**、*斜体*、[链接](https://example.com)、代码块、表格。

![图片说明](/images/my-figure.png)
```

4. 点 **Commit changes** → 等一两分钟，Actions 跑完文章就上线了。

- `title` / `date` / `category` 必填；`date` 必须是 `YYYY-MM-DD`
- `categories` 支持用 `/` 表示**子文件夹层级**，比如
  `Course notes/Computer Vision/Computer Vision III` 会在博客页左侧生成
  三层嵌套的文件夹；点任意一层都能筛出该层级下的所有文章
- 一篇文章也可以属于多个文件夹：`categories: [A/B, C]`
- `tags` 随意多个，显示为小标签且可被搜索

## 插图

1. 仓库页面 → `images/` 目录 → **Add file → Upload files**，把图片拖进去提交
2. 正文里写 `![说明](/images/文件名.png)`

## 修改 / 删除文章

- 修改：直接在网页上编辑对应的 `.md` 文件（铅笔图标），提交即可
- 删除：删掉 `.md` 文件，Actions 会连生成的页面一起清掉

## 本地预览（可选）

```bash
pip install markdown pyyaml
python3 scripts/build_posts.py
```
