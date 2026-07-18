# 🗺️ Roadmap: Track ML Experiments (Git + DVC)

Yeh roadmap aapko batayega ke jab bhi aap koi parameter change karte hain toh usko **track (save)** kaise karna hai taake aap kabhi bhi purani state pe wapas ja sakein.

## 🔄 The Golden Cycle (Har experiment ke liye yeh 3 steps karein)

Aapne abhi `params.yaml` mein changes kiye hain:
- `test_size: 0.22`
- `max_features: 45`
- `n_estimators: 26`

Ab is naye experiment ko hamesha ke liye track karne ka roadmap yeh hai:

### Step 1: Run the Pipeline (DVC ko apna kaam karne dein)
Jab aap parameters change karein, toh pipeline chalayen. DVC automatically un files ko dubara chalayega jinhe zaroorat hai.

```bash
dvc repro
```
> [!NOTE] 
> DVC automatically naye results calculate karke unka MD5 hash `dvc.lock` mein update kar dega aur naye data ko `.dvc/cache` mein save kar lega.

### Step 2: Track Changes with Git
Ab jab DVC ne `dvc.lock` file update kar di hai, toh humein Git ko batana hai ke yeh ek **naya version / experiment** hai isko save kar lo.

```bash
git add params.yaml dvc.lock
git commit -m "Experiment: test_size=0.22, max_features=45, n_estimators=26"
```
> [!IMPORTANT]
> Hamesha **`params.yaml`** aur **`dvc.lock`** dono ko commit karein! `params.yaml` aapki values save karta hai, aur `dvc.lock` aapke data ka address save karta hai.

### Step 3: Push to Safe Storage (Optional but Recommended)
Apne code ko GitHub pe aur data ko remote storage (jaise S3 ya GDrive) pe bhej dein taake kuch lose na ho.

```bash
git push origin main
dvc push
```
*(Note: `dvc push` tab chalega jab aapne remote S3 setup kiya ho).*

---

## 🕰️ Time Machine (Purane Experiment Pe Wapas Jana)

Agar naya experiment acha nahi chala aur aapko purane wale par jana hai, toh yeh roadmap follow karein:

### Step 1: Purane Experiments Ki List Dekhein
```bash
git log --oneline
```
Output kuch aisa hoga:
```
abc1234 Experiment: test_size=0.22, max_features=45
def5678 Integrated params.yaml into pipeline
```

### Step 2: Code wapas layen (Git)
Man lein aapko `def5678` wale experiment par wapas jana hai:
```bash
git checkout def5678 -- params.yaml dvc.lock
```

### Step 3: Data wapas layen (DVC)
Ab DVC ko bolein ke jo `dvc.lock` aapne mangwai hai, us hisab se data ko reset kar de:
```bash
dvc checkout
```
> [!TIP]
> That's it! Bina kuch run kiye aapka purana data aur model wapas aa jayega.

---

## 🔍 Best Practices Checklist
- `[x]` Sirf `params.yaml` mein values change karein.
- `[x]` Python code (`.py` files) mein hardcoded values na likhein.
- `[x]` Har experiment ke baad ek descriptive Git commit hamesha karein.
- `[x]` Kabhi bhi data folders (`data/` ya `models/`) ko directly delete ya rename na karein, unhe DVC ke hawalay rehne dein.
