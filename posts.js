/* ---------------------------------------------------------------------------
   Blog index — the single source of truth for the post list.

   To publish a new post:
     1. Create the page under  posts/<slug>.html  (copy an existing one).
     2. Add an entry to the POSTS array below.
   Nothing on the site can add posts; the list only changes when this file is
   committed to the repository.

   Fields:
     title    string   shown in the list and used for name sorting
     date     string   ISO "YYYY-MM-DD" — used for date sorting
     category string   one filter chip is generated per distinct category
     tags     string[] optional, searchable
     url      string   path to the post page
     summary  string   one or two lines shown under the title, searchable
--------------------------------------------------------------------------- */

const POSTS = [
  {
    title: "Robot Learning — lecture notes",
    date: "2026-08-20",
    category: "Course notes",
    tags: ["robot learning", "imitation learning", "reinforcement learning", "TUM"],
    url: "posts/robot-learning.html",
    summary: "Running notes from the TUM Robot Learning lecture — policy representations, imitation learning, and model-based control."
  },
  {
    title: "Hello, and what this blog is for",
    date: "2026-08-23",
    category: "Meta",
    tags: ["about"],
    url: "posts/hello.html",
    summary: "Why I keep these notes in public, and how the archive is organised."
  }
];
