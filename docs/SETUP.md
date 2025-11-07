# Documentation Website Setup

This folder contains the Jekyll-based documentation website for the Attention Is All You Need project.

## Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── _layouts/            # Custom layouts
│   ├── default.html    # Default page layout
│   └── home.html       # Homepage layout
├── index.md            # Homepage (uses home.html layout)
├── architecture.md     # Architecture documentation
├── training.md         # Training guide
├── api.md             # API reference
├── examples.md        # Examples and tutorials
├── favicon.svg        # Site favicon (🦢)
├── Gemfile            # Ruby dependencies
├── README.md          # Documentation index
└── SETUP.md           # This file

Generated (excluded from git):
├── _site/             # Built website
└── .jekyll-cache/     # Build cache
```

## Local Development

```bash
# Install dependencies (first time only)
bundle install

# Serve locally with auto-reload
bundle exec jekyll serve

# Visit http://localhost:4000
```

## Design Features

- **Modern UI**: Clean, responsive design with custom layouts
- **Swan Favicon**: 🦢 emoji as the site icon
- **Typography**: Inter for body text, JetBrains Mono for code
- **Interactive**: Hover effects on cards and links
- **Mobile-friendly**: Responsive navigation and layout

## Customization

The site uses custom Jekyll layouts instead of the default Cayman theme:
- `_layouts/home.html` - Hero section, feature cards, quick start
- `_layouts/default.html` - Standard content pages with navigation

All styles are embedded in the layouts for simplicity.

## Deployment

GitHub Pages automatically builds and deploys the site from the `docs/` folder on the main branch.

## Prerequisites

- GitHub repository: `1AyaNabil1/attention_is_all_you_need`
- Documentation files in the `docs/` folder
- GitHub account with repository access

---

## Quick Setup

### Step 1: Enable GitHub Pages

1. Go to your repository on GitHub: https://github.com/1AyaNabil1/attention_is_all_you_need
2. Click on **Settings** (in the top navigation bar)
3. Scroll down to the **Pages** section (left sidebar under "Code and automation")
4. Under **Source**, select:
   - **Branch**: `main` (or your default branch)
   - **Folder**: `/docs`
5. Click **Save**

### Step 2: Wait for Deployment

GitHub will automatically build and deploy your site. This typically takes 1-3 minutes.

You can check the deployment status:
- Go to **Actions** tab in your repository
- Look for the "pages-build-deployment" workflow

### Step 3: Access Your Documentation

Once deployed, your documentation will be available at:

```
https://1ayanabil1.github.io/attention_is_all_you_need/
```

---

## Customization

### Theme Configuration

The documentation uses the **Cayman** theme. You can customize it by editing `docs/_config.yml`:

```yaml
# Current configuration
title: Attention Is All You Need - PyTorch Implementation
description: A clean, well-documented PyTorch implementation of the Transformer model
theme: jekyll-theme-cayman
```

### Available Jekyll Themes

You can change the theme by modifying the `theme` field:

- `jekyll-theme-architect`
- `jekyll-theme-cayman` (current)
- `jekyll-theme-dinky`
- `jekyll-theme-hacker`
- `jekyll-theme-leap-day`
- `jekyll-theme-merlot`
- `jekyll-theme-midnight`
- `jekyll-theme-minimal`
- `jekyll-theme-modernist`
- `jekyll-theme-slate`
- `jekyll-theme-tactile`
- `jekyll-theme-time-machine`

### Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file to the `docs/` folder with your domain:
   ```
   docs.yourdomain.com
   ```

2. Configure DNS with your domain provider:
   - Add a CNAME record pointing to `1ayanabil1.github.io`

3. Update GitHub Pages settings:
   - Go to Settings → Pages
   - Enter your custom domain
   - Enable "Enforce HTTPS"

---

## File Structure

Your current documentation structure:

```
docs/
├── _config.yml          # Jekyll configuration
├── index.md            # Home page
├── architecture.md     # Architecture documentation
├── training.md         # Training guide
├── api.md             # API reference
├── examples.md        # Examples and tutorials
└── README.md          # Documentation README
```

---

## Verification Checklist

After enabling GitHub Pages, verify:

- [ ] Pages is enabled in repository settings
- [ ] Source is set to `main` branch and `/docs` folder
- [ ] Build workflow completed successfully in Actions tab
- [ ] Site is accessible at the GitHub Pages URL
- [ ] All internal links work correctly
- [ ] Images (if any) load properly
- [ ] Code blocks render correctly
- [ ] Navigation between pages works

---

## Troubleshooting

### Issue: Pages not building

**Solution:**
- Check the Actions tab for build errors
- Ensure all Markdown files have valid syntax
- Verify `_config.yml` is valid YAML

### Issue: 404 errors

**Solution:**
- Ensure the homepage is named `index.md`
- Check that all internal links use relative paths
- Wait a few minutes after enabling Pages

### Issue: Styles not loading

**Solution:**
- Clear your browser cache
- Verify the theme name in `_config.yml`
- Check that Jekyll processed the theme correctly

### Issue: Build fails with theme error

**Solution:**
- Use a supported GitHub Pages theme
- Check theme name spelling
- Try switching to `theme: jekyll-theme-minimal` temporarily

---

## Local Testing

Test your documentation locally before deploying:

### Option 1: Python HTTP Server

```bash
cd docs
python3 -m http.server 8000
```

Visit: http://localhost:8000

**Note:** This won't process Jekyll templates, just serve static files.

### Option 2: Jekyll (Recommended)

```bash
# Install Jekyll (one-time setup)
gem install bundler jekyll

# Create Gemfile in docs/ folder
cd docs
cat > Gemfile << EOF
source "https://rubygems.org"
gem "github-pages", group: :jekyll_plugins
EOF

# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve
```

Visit: http://localhost:4000

This will process Jekyll templates exactly like GitHub Pages.

---

## Analytics (Optional)

Add Google Analytics to track visitors:

1. Get your Google Analytics tracking ID
2. Add to `_config.yml`:
   ```yaml
   google_analytics: UA-XXXXXXXXX-X
   ```

---

## Security

### HTTPS

GitHub Pages automatically provides HTTPS for `.github.io` domains. To enforce HTTPS:

1. Go to Settings → Pages
2. Check "Enforce HTTPS"

### Security Headers

GitHub Pages automatically adds appropriate security headers.

---

## Monitoring

### GitHub Actions

Monitor your Pages deployment:
- Go to **Actions** tab
- View "pages-build-deployment" workflow
- Check build logs for errors

### Status Badge

Add a deployment status badge to your README:

```markdown
[![GitHub Pages](https://github.com/1AyaNabil1/attention_is_all_you_need/workflows/pages-build-deployment/badge.svg)](https://github.com/1AyaNabil1/attention_is_all_you_need/actions/workflows/pages-build-deployment)
```

---

## Updating Documentation

To update your documentation:

1. Edit files in the `docs/` folder
2. Commit and push to the `main` branch
3. GitHub will automatically rebuild and redeploy
4. Wait 1-3 minutes for changes to appear

### Quick Update Workflow

```bash
# Make changes to documentation
cd docs
vim index.md

# Commit and push
git add .
git commit -m "Update documentation"
git push origin main

# GitHub Actions will automatically rebuild
```

---

## Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Jekyll Themes](https://pages.github.com/themes/)
- [GitHub Pages Examples](https://github.com/collections/github-pages-examples)

---

## Tips

1. **Use Relative Links**: Always use relative paths for internal links
   ```markdown
   [Architecture](architecture.md)  ✅
   [Architecture](/architecture)    ❌
   ```

2. **Preview Changes**: Test locally before pushing
3. **Keep it Simple**: Stick to standard Markdown features
4. **Optimize Images**: Compress images before adding them
5. **Use Anchors**: Add section anchors for deep linking
   ```markdown
   ## My Section {#my-section}
   ```

---

## Success

Once everything is set up, your documentation should be live at:

**https://1ayanabil1.github.io/attention_is_all_you_need/**

Share this URL in your main README.md and with users who want to learn more about your Transformer implementation!

---

<div align="center">
  <p><em>Implemented by AyaNexus 🦢</em></p>
</div>
