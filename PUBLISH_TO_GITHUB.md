# Publishing this repository on GitHub (about 5 minutes)

This folder is already a complete git repository with its history. You only need to create an empty repository on GitHub and push to it.

## 1. Create the empty repository on GitHub
1. Sign in at https://github.com and open https://github.com/new
2. Repository name: `vae-palns-shift`
3. Choose **Public**, or **Private** if you only want to share it with the editor and reviewers for now.
4. Leave all initialization options **unticked** (no README, no .gitignore, no licence), because the repository already has them.
5. Click **Create repository**.

## 2. Set your GitHub username and push
Open Terminal and run the commands below, replacing `YOUR_GITHUB_USERNAME`:

```bash
cd "/Volumes/TOSHIBA_4IB/01_Resarch/Research in Progress/Evolutionary Intelligence/JoH_LearnedGuidance_ALNS/03_Code_and_data/vae-palns-shift"
./set_github_user.sh YOUR_GITHUB_USERNAME        # replaces USERNAME in CITATION.cff and commits the change
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/vae-palns-shift.git
git push -u origin main
```

When git asks for a password, use a personal access token (GitHub → Settings → Developer settings → Personal access tokens), not your account password. Alternatively, use GitHub Desktop: **File → Add local repository**, choose this folder, then click **Publish repository**.

## 3. Update the manuscript
In `01_Submission_JoH` and `02_LaTeX_source`, replace `USERNAME` with your GitHub username in `manuscript.tex`, `title_page.tex` and `cover_letter.tex`, then recompile, or ask Claude to do it.

## 4. Archive on Zenodo (on acceptance, for a permanent DOI)
1. Sign in at https://zenodo.org with your GitHub account, then go to **GitHub** in the account menu and switch the repository on.
2. On GitHub, create a release (for example `v1.0.0`). Zenodo archives it automatically and assigns a DOI.
3. Add the DOI to the paper's data-availability statement and to `CITATION.cff`.

## Notes
- The repository is about 55 MB. The largest file is well under GitHub's 100 MB limit, so Git LFS is not needed.
- The Lima/Callao district polygons are deliberately excluded because their licence is unclear (see `data/LICENSES.md`). Everything else needed to reproduce the paper is included.
