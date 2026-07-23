# Training the VC-GAN on Google Colab

Local machines here have no GPU (TF ≥ 2.11 has none on native Windows), so train
on a free Colab T4. You only need to train once — download `generator.weights.h5`
and run `infer.py` locally.

**Runtime → Change runtime type → T4 GPU** first.

### 1. Install + get the code

```python
!pip -q install qrcode[pil] pycryptodome
# upload vcgan.py and cryptography.py (Files panel) OR clone your repo:
# !git clone <your-repo-url> && %cd <repo>
```

### 2. Get the landscape dataset (Kaggle)

Upload your `kaggle.json` (Kaggle → Account → Create New API Token), then:

```python
from google.colab import files; files.upload()          # pick kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip -q install kaggle
!kaggle datasets download -d arnaud58/landscape-pictures -p landscapes --unzip
```

### 3. Train

```python
import vcgan
G = vcgan.train("landscapes", epochs=200, batch=16)      # ~a few hours on T4
```

Watch the logs: `rec` (reconstruction) should fall — that means the overlay is
learning to reveal the QR. `con` (content) staying low means shares still look
like their landscapes. If `rec` stalls high, raise `vcgan.LAMBDA_RECON` (top of
`vcgan.py`) and retrain — that is the beauty-vs-recovery knob.

### 4. Download the weights

```python
from google.colab import files
files.download("generator.weights.h5")
```

Drop `generator.weights.h5` next to `infer.py` locally, then:

```bash
python infer.py cover1.jpg cover2.jpg
```

to generate + verify shares for a real signed transaction.
