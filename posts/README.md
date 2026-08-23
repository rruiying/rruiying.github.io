# Adding a blog post

The blog is static. There is no form, no CMS, and no way to publish from the
website — a post only appears after you commit these two changes.

## 1. Create the page

```bash
cp posts/_template.html posts/my-new-post.html
```

Then edit the copy: replace `POST TITLE` (in `<title>` and `<h1>`), the date,
the category in `<span class="post-category">`, and the body.

`_template.html` starts with an underscore and is never listed in the blog
index, so it is safe to leave in place.

## 2. Register it in `posts.js`

Add an object to the `POSTS` array:

```js
{
  title: "My New Post",
  date: "2026-09-01",          // ISO, drives the date sorting
  category: "Course notes",    // a filter chip is created per distinct category
  tags: ["slam", "3d vision"], // optional, searchable
  url: "posts/my-new-post.html",
  image: "images/blog/my-cover.jpg",  // optional card cover; omit for a text card
  summary: "One or two lines, shown on the card and searchable."
}
```

## Images

- **Card cover:** put the file under `images/blog/` and set the `image` field.
  Landscape around 3:2 looks best; the card crops with `object-fit: cover`.
- **Inside a post:** use a `<figure>` (note the `../` prefix, since post pages
  live one level down):

  ```html
  <figure>
      <img src="../images/blog/my-figure.jpg" alt="">
      <figcaption>Caption text.</figcaption>
  </figure>
  ```

Order in the file does not matter — `blog.html` sorts on load.

## Categories

Whatever strings you use become the filter chips automatically. Keeping them to
a small set works best, e.g. `Course notes`, `Paper notes`, `Projects`, `Meta`.
