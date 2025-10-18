# GitHub SSH Setup Guide

## Generate SSH Key

```bash
ssh-keygen -t ed25519 -C "juancamilomabe@gmail.com"
```

- Press Enter to accept default location (~/.ssh/id_ed25519)
- Enter a passphrase (or press Enter for no passphrase)

## Add SSH Key to ssh-agent

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## Add SSH Key to GitHub

1. Copy your public key:
```bash
cat ~/.ssh/id_ed25519.pub | pbcopy
```

2. Go to: https://github.com/settings/ssh/new
3. Title: "MacBook" (or whatever you prefer)
4. Key: Paste (Cmd+V) your key
5. Click "Add SSH key"

## Test Connection

```bash
ssh -T git@github.com
```

You should see: "Hi camilojourney! You've successfully authenticated..."

## Then Push Your Repo

```bash
git remote add origin git@github.com:camilojourney/pronunciation-error-detection.git
git push -u origin main
```

---

**Which method do you prefer?**
- HTTPS (simpler, need token each time)
- SSH (one-time setup, more secure, no passwords)
