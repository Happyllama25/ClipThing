# ClipThing - ALPHA

### A cross-platform game clips tool for managing, trimming, and exporting game clips.
### Zero accounts, Zero ads, Free, Forever[^1]

#### Major features:
- Multi audio channel support
    - Edit volumes for each audio channel individually and combines them into one
- Sharing under a specified size limit
- Saving of trimmed settings
- Saving of exported settings

### Getting Started
lorem ipsum i forgot

run `python clipthing.py` in a terminal, but run `pip install -r requirements.txt` first.

Runs on `http://localhost:8000`


You can optionally host the program on a remote server/NAS so as to not use your hardware during a game session <-- self note: add this selling point somewhere where it makes sense

> [!WARNING]
> This program is meant to be used in a local network - there is no remote authentication currently! Tailscale and alternative users should be fine if you know what you are doing

### Known (probably) issues:

- if you export the same clip twice (or more) with  different settings, the download link will only download whichever clip has finished processing most recently
    - as in, the first export click will be downloaded, instead of the second export click, because the second click hasn't finished processing yet
        - (unless the title changes, if the title changes, ur good [i think])
- if you delete the local file, the database wont know. https://github.com/users/Happyllama25/projects/2/views/1?pane=issue&itemId=133228518


im so great at this (/s)

### ✨ CONTRIBUTING ✨

please.

[^1]:  (as long as GitHub still exists)