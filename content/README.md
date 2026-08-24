# 怎么发博客

每篇文章 = `content/posts/` 里的一个 Markdown 文件。**文件放在哪个文件夹，
博客页左侧的分类树就长什么样** —— 不用在 front matter 里写分类。

```
content/posts/Course_notes/Computer_Graphics/Animation/Games103/games103-note1.md
                └──────── 这个目录路径就是文章的分类：
                          Course notes › Computer Graphics › Animation › Games103
```

- 目录名里的下划线显示时自动变空格，`(IN2064)` 前自动补空格
- 新分类 = 新建文件夹，放进第一篇文章即可
- **文件名 = 文章网址**，用小写英文加连字符（`games103-note1.md`），
  发布后不要改名（会断链接），全站文件名不能重复（构建会报错拦住）

## front matter 模板

```markdown
---
title: "标题（含冒号必须加引号）"
date: 2026-09-01
tags: [physics simulation, GAMES103]
summary: 一两句摘要，显示在列表里、参与搜索。注意整行不能出现英文冒号。
---
正文 Markdown。公式用 $...$（行内）和 $$...$$（独立）。
```

`title` 和 `date` 必填；`tags`、`summary` 建议写。

## 插图

图片放进 `images/blog/` 下对应的文件夹（建议和文章目录结构一致），正文里：

```markdown
![图注文字](/images/blog/Course_notes/Computer_Graphics/Games103/note1/xxx.png)
```

图注（方括号里的字）会显示在图片下方。构建时自动写入图片尺寸、懒加载；
引用了不存在的图片会在构建日志里警告。
**图片路径里不要出现括号和空格**（会破坏 Markdown 链接）。

## 修改 / 删除

- 修改：编辑 md 文件，提交即可（"Updated" 日期自动取 git 记录）
- 删除：删掉 md 文件，生成的页面会被自动清理
- `posts/`、`posts.js`、`sitemap.xml` 全部是机器生成的，永远不要手改

## 本地预览

```bash
pip install markdown pyyaml pymdown-extensions pillow
python3 scripts/build_posts.py
```
