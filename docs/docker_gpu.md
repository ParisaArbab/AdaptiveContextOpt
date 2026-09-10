# اجرای Docker روی A100

ترتیب pipeline تغییر نمی‌کند: Graphify → LeanCTX واقعی → feedback →
localization/evaluation. سرویس `ollama` مدل را روی GPU اجرا می‌کند؛ سرویس
`pipeline` ابزارها، تست‌های بنچمارک و ارزیابی را اجرا می‌کند.

## پیش‌نیاز سرور

Linux، Docker Engine و Compose v2 با پشتیبانی `up --wait`، درایور NVIDIA،
و [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
نصب درایور و پیکربندی runtime باید روی میزبان انجام شود؛ image آن‌ها را
جایگزین نمی‌کند. تنظیم GPU مطابق
[مستندات Docker](https://docs.docker.com/compose/how-tos/gpu-support/) است.

```bash
nvidia-smi
docker info
docker compose version
```

## انتقال و اجرای نمونه

نسخهٔ فعلی پروژه، شامل فایل‌های جدید Docker و اسکریپت‌ها را به سرور منتقل کنید.
برای جلوگیری از انتقال باینری‌ها و virtualenvهای مک:

```bash
rsync -av --exclude='.git' --exclude='.tools' --exclude='.venv' \
  --exclude='data/repos' --exclude='results' --exclude='references' \
  --exclude='graphify-out' --exclude='__pycache__' \
  ./ USER@SERVER:~/AdaptiveContextOpt/
```

روی سرور، داخل پوشهٔ پروژه:

```bash
PIPELINE_DEVICE=nvidia bash scripts/run_docker.sh
```

این دستور image را می‌سازد، Ollama را روی GPU شمارهٔ ۰ راه‌اندازی می‌کند،
در صورت نبود مدل `llama3.1:8b` آن را دانلود می‌کند و یک instance
`sympy__sympy-17139` را با `full,pure_flexfl` اجرا می‌کند. benchmark checkout
و مدل در volumeهای پایدار می‌مانند. خروجی هر اجرا در
`results/docker_<timestamp>/` قرار می‌گیرد.

اجرای بدون پارامتر از command پیش‌فرض Compose استفاده می‌کند. وقتی پارامتر
می‌دهید، تمام تنظیمات موردنیاز اجرای خود را وارد کنید؛ command پیش‌فرض جایگزین می‌شود:

```bash
PIPELINE_DEVICE=nvidia MODEL=llama3.1:8b GPU_DEVICE=0 \
  bash scripts/run_docker.sh \
  --instances /inputs/smoke_sympy_17139.json \
  --arms full,pure_flexfl --limit 1
```

برای ورودی کامل **نرمال‌شدهٔ همین پروژه**:

```bash
PIPELINE_DEVICE=nvidia bash scripts/run_docker.sh \
  --instances /inputs/instances.json \
  --dataset swe-bench-lite --arms full,pure_flexfl --limit 0
```

`/inputs` همان پوشهٔ `data/` میزبان است و فقط خواندنی mount می‌شود. برای
فایل خام Hugging Face از fetcher پروژه استفاده کنید؛ تغییر نام فایل خام
به `instances.json` آن را نرمال نمی‌کند. برای fetch خودکار همهٔ ۳۰۰ نمونه:

```bash
PIPELINE_DEVICE=nvidia bash scripts/run_docker.sh \
  --dataset swe-bench-lite --n 300 --arms full,pure_flexfl --limit 0
```

## مدل‌ها و بنچمارک‌های موجود

برای استفاده از مدل محلی موجود، پوشهٔ `.ollama` شامل `models/blobs` و
`models/manifests` را به سرور منتقل کنید و مسیر آن را تنظیم کنید:

```bash
export OLLAMA_MODELS_DIR=/absolute/path/to/copied-dot-ollama
export INPUTS_DIR=/absolute/path/to/benchmark-json-directory
export RESULTS_DIR=/absolute/path/to/results
PIPELINE_DEVICE=nvidia bash scripts/run_docker.sh \
  --instances /inputs/instances.json --arms full,pure_flexfl --limit 1
```

اگر مدل با نام دقیق `MODEL` در storage باشد، دوباره دانلود نمی‌شود.
برای checkoutهای از قبل آمادهٔ لینوکس می‌توان `BENCHMARK_REPOS_DIR` را
به پوشه‌ای با ساختار `<instance_id>/` تنظیم کرد. virtualenv و graph cache
ساخته‌شده روی مک را به این مسیر منتقل نکنید؛ آن‌ها را در لینوکس بازسازی کنید.
checkoutها محیط کاری مدیریت‌شدهٔ harness هستند و reset/clean می‌شوند.

## معماری و نسخه‌ها

- Docker معماری لینوکس میزبان را انتخاب می‌کند؛ `platform: amd64` تحمیل نشده است.
- Python 3.11 روی Debian Bookworm است. pip wheel مناسب همان معماری را انتخاب می‌کند.
- Graphify و چند dependency آزموده‌شده در `docker/constraints.txt` ثابت‌اند؛
  نسخهٔ تمام dependencyهای resolved در image و `environment.json` ذخیره می‌شود.
  dependencyهای transitively resolved و image پایه lock کامل ندارند؛ برای
  تکرار دقیق همان محیط، image ساخته‌شده را با `docker save/load` منتقل کنید.
- LeanCTX رسمی 3.10.1 هنگام build داخل لینوکس نصب و SHA256 آن تأیید می‌شود.
  فایل اجرایی مک وارد image نمی‌شود. این مسیر shell compression به CUDA نیاز ندارد.
- Ollama روی نسخهٔ 0.33.3 تنظیم شده است؛ `OLLAMA_IMAGE` اجازهٔ انتخاب tag/digest دیگر می‌دهد.
- حالت `PIPELINE_DEVICE=auto` روی Linux دارای NVIDIA حالت GPU و در سایر
  محیط‌ها CPU را انتخاب می‌کند. برای A100 صریحاً `nvidia` را استفاده کنید.
- Docker روی مک در این تنظیمات از GPU اپل استفاده نمی‌کند. برای تست CPU:
  `PIPELINE_DEVICE=cpu bash scripts/run_docker.sh`.

## صحت اجرا و لاگ‌ها

برای اجرای تست‌های رگرسیون داخل محیط لینوکسی image، قبل از benchmark:

```bash
docker build --target test -t adaptive-context-pipeline:test .
```

مرحلهٔ test تمام تست‌ها، از جمله تست LeanCTX واقعی، را اجرا می‌کند و در صورت
شکست build ناموفق می‌شود. image معمولی Compose از مرحلهٔ `runtime` ساخته می‌شود.

قبل از benchmark، `nvidia-smi` داخل کانتینر و سپس allocation واقعی VRAM
برای مدل بررسی می‌شود. GPU mode بدون allocation متوقف می‌شود. این بررسی
استفاده از GPU را تأیید می‌کند؛ به معنی تضمین offload کامل تمام لایه‌ها نیست.

```bash
docker compose -f compose.yaml -f compose.nvidia.yaml exec ollama ollama ps
docker compose -f compose.yaml -f compose.nvidia.yaml logs ollama
```

هر خروجی شامل `environment.json` با نسخهٔ Ollama، اطلاعات مدل، allocation،
dependencyها، command و خطاهای راه‌اندازی است. لاگ‌ها و artifactهای تمام
مراحل طبق `smoke_run_logging.md` ذخیره می‌شوند. warm-up مدل قبل از benchmark
انجام می‌شود و جزو شمارش comparative pipeline نیست. اجرای دارای skipped
instance/arm با exit ناموفق برمی‌گردد.

این Docker محیط خود pipeline را ایزوله می‌کند. تست‌های بنچمارک با مسیر
`--local-fallback` **داخل همین کانتینر** اجرا می‌شوند؛ این کار معادل image
رسمی اختصاصی SWE-bench برای هر instance نیست. نسخه‌های خاص موردنیاز برخی
ریپوهای قدیمی همچنان ممکن است به محیط اختصاصی نیاز داشته باشند. Docker
socket میزبان به pipeline mount نشده است.

برای توقف سرویس بدون حذف مدل‌ها/checkoutها:

```bash
docker compose -f compose.yaml -f compose.nvidia.yaml down
```

از `down -v` برای توقف معمولی استفاده نکنید؛ volumeهای پایدار را حذف می‌کند.
روی Linux فایل خروجی ممکن است متعلق به root کانتینر باشد؛ در صورت نیاز مالکیت
پوشهٔ خروجی را روی میزبان به کاربر خود منتقل کنید.
