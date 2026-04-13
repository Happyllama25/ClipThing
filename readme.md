# ClipThing - ALPHA

### A cross-platform game clips tool for managing, trimming, and exporting game clips.
### Zero accounts, Zero ads, Free, Forever[^1]

#### Major features:
- Multi audio channel support (the main motivator for this project)
    - Edit volumes for each audio channel individually and combines them into one
- Exporting under a specified size limit
- API based (you can make your own UI or interface!) (oh crap this means i need to write documentation)

### Getting Started
lorem ipsum i forgot

`pip install -r requirements.txt` (guh)
`python ClipThing.py` - you can run `-nogui` to skip the system tray icon

Runs on port `8000`, normally accessible on `http://localhost:8000`

`CLIPS_ROOT` environment variable is the root directory for scanning and storing clips and data - defaults to `~HOME/Videos/clips`
##### note that on MacOS this creates a new Videos directory in the users Home, because MacOS uses Movies instead of Videos

Goal is to pack this into a neat lil' Docker container for running on a NAS, not quite there yet

> [!WARNING]
> This program is meant to be used in a local network - there is no authentication! This should not be exposed to the internet (dont port forward). Tailscale and alternative users should be fine if you know what you are doing


im so great at this (/s)

### ✨ CONTRIBUTING ✨

please.

look i know I KNOW, its monolithic, its ugly, its not modularised, and its not pretty (thats debatable - i for one think its beautiful).

BUT

it works. AND, you found it, and were interested in it enough to read this section, so thats a win in my book.

[^1]:  (as long as GitHub still exists)
