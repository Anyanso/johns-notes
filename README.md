# CMPT 423 Study Hub

A static website for the course: midterm prep, study notes (Lectures 02–19) and a plain-language "Explain it simply" dropdown inside every lecture.

Live site (after step 3 below): `https://<your-github-username>.github.io/<repo-name>/`

## Publish it (once)

1. Create a new repository on GitHub and push this folder to it (`git init`, `git add .`, `git commit`, `git remote add origin ...`, `git push -u origin main`).
2. On GitHub open **Settings → Pages**. Under **Build and deployment** choose **Deploy from a branch**, branch **main**, folder **/docs**. Save.
3. After a minute the link above works. Send it to your team.

GitHub Pages on a free account needs a **public** repository, so anyone with the link can read the notes. Do not put assignment solutions in the repo (the course allows GenAI only for writing clarity, and posting solutions is academic misconduct).

## Update it (every time something changes)

Edit the source text, rebuild, push.

| To change | Edit |
|---|---|
| Lecture notes, glossary, formula sheet, code, errata | `content/study_notes.md` |
| The plain-language dropdown for each lecture | `content/easy_edition.md` (same lecture headings: `## Lecture 18 — ...`) |
| Midterm prep (plan, lessons, questions, mock exam) | `tools/prep/content.py` and `tools/prep/page.html` |

Then:

```bash
cd tools && npm install        # first time only
cd .. && python3 tools/build_all.py
git add -A && git commit -m "Add lecture 20" && git push
```

`build_all.py` needs Python 3 with `markdown` and `pygments` (`pip install markdown pygments`) and Node.js. It regenerates `docs/index.html`, `docs/notes.html` and `docs/prep.html`; GitHub Pages serves the `docs/` folder.

## Adding a new lecture

1. Add a section `## Lecture 20 — Title` to `content/study_notes.md`, with `### 20.1 ...` subsections, then `#### 📖 Textbook deep dive`, `#### ➕ Textbook extra` and `#### ✅ Key takeaways`, as in the existing lectures.
2. Add a matching `## Lecture 20 — Title` section to `content/easy_edition.md`. Everything before `#### 📖 Textbook deep dive` becomes the dropdown.
3. Rebuild and push.

Math is written as `$...$` (inline) and `$$...$$` (display) and is rendered when you build. Highlights, notes and drawings made on the site are saved in each person's own browser; they are not shared.
