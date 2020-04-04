<div align="center"><h1>📝 Dissertation</h1></div>

[![pipeline status](https://gitlab.com/harpocrates-app/dissertation/badges/master/pipeline.svg)](https://gitlab.com/harpocrates-app/dissertation/-/commits/master)
[![made-with-latex](https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg)](https://www.latex-project.org/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

This repository contains the LaTeX code for my dissertation that is auto compiled on each push to master in Gitlab CI, the up to date final output PDF is [available here](https://harpocrates-app.gitlab.io/dissertation/dissertation.pdf).

Development (issues, boards, CI, Docker respositories...) happens on [gitlab.com](https://gitlab.com/harpocrates-app/dissertation) and a mirror (source code only, no issues etc...) of this repository is available on [github.com](https://github.com/guillaumedsde/dissertation).

<div align="center">
    <a href="https://harpocrates-app.gitlab.io/dissertation/dissertation.pdf">
        <img width="50%" src="https://harpocrates-app.gitlab.io/dissertation/dissertation.jpg" alt="Rendered preview">
    </a>
</div>


## LaTeX

This template has been tested with `latexmk`. It should also work with `xelatex` and `lualatex`. Note that on Linux you may need to copy the contents of the `fonts/` folder to `~/.fonts/`.