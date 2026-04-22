# HACKATHON DEPLOYMENT GUIDE — Follow Every Step Exactly

## STEP 1: Get Gemini API Key (2 minutes)

1. Go to: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Select any Google Cloud project (or create one — it's instant)
5. COPY the API key and save it somewhere — you'll need it in Step 3

---

## STEP 2: Push Code to GitHub (5 minutes)

### If you DON'T have Git installed:
1. Go to https://github.com → Sign in → Click "+" → "New repository"
2. Name it: `hackathon-api`
3. Keep it Public, DON'T add README
4. Click "Create repository"
5. On the next page, click "uploading an existing file"
6. Drag and drop ALL 5 files from the project folder:
   - main.py
   - requirements.txt
   - Procfile
   - railway.json
   - .gitignore
7. Click "Commit changes"

### If you HAVE Git installed (faster):
```bash
cd hackathon-api
git init
git add .
git commit -m "hackathon api"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hackathon-api.git
git push -u origin main
```

---

## STEP 3: Deploy on Railway (5 minutes)

1. Go to: https://railway.com
2. Click "Login" → Sign in with GitHub
3. Click "New Project"
4. Click "Deploy from GitHub Repo"
5. Select your `hackathon-api` repository
6. Railway will auto-detect Python and start building
7. While it builds, click on the service → "Variables" tab
8. Click "New Variable":
   - Name: `GEMINI_API_KEY`
   - Value: paste your Gemini API key from Step 1
9. Click "Add"
10. Go to "Settings" tab → "Networking" section
11. Click "Generate Domain" — this gives you a public URL like:
    `https://hackathon-api-production-xxxx.up.railway.app`
12. WAIT for deployment to finish (green checkmark)

---

## STEP 4: Test Your API (2 minutes)

Open your terminal/CMD and run:

```bash
curl -X POST https://YOUR-RAILWAY-URL/api/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 10 + 15?", "assets": []}'
```

On Windows CMD:
```
curl -X POST https://YOUR-RAILWAY-URL/api/answer -H "Content-Type: application/json" -d "{\"query\": \"What is 10 + 15?\", \"assets\": []}"
```

Expected response: {"output": "The sum is 25."}

If it says "The sum is 25." or something very close — YOU'RE GOOD.

---

## STEP 5: Submit to the Challenge Platform

Your API URL to submit is:
```
https://YOUR-RAILWAY-URL/api/answer
```

Paste this in the "SUBMIT YOUR API" field and hit Submit.

---

## TROUBLESHOOTING

**"Application failed to respond"** → Check Railway logs (click on service → "Logs")
**"502 Bad Gateway"** → Deployment still in progress, wait 30 seconds
**Wrong answer format** → Come back to me, I'll adjust the prompt
**Gemini rate limit** → Free tier is 15 req/min, should be fine

---

## FOR FUTURE LEVELS

After passing Level 1, come back and tell me what Level 2's test case looks like.
I'll adjust the system prompt or add capabilities (PDF reading, image analysis, etc.)

The code already handles:
- Asset URL fetching (for when they send documents/data)
- Flexible response generation
- Error handling with fallbacks
