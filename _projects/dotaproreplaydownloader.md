---
layout: page
title: Dota Pro Replay Downloader
description: Application that downloads 'pro-level' Dota 2 replays of the specified hero to the Dota 2 replays directory.
img: assets/img/proj6.png
importance: 3
category: independent
---

The GitHub repository for this project is located [here](https://github.com/WHDaniels/dotaProReplayDownloader).

## Summary
This program takes data in the form of match ID's of high MMR players from [dota2protracker.com](https://www.dota2protracker.com/) 
for the hero the user has specified. The replays of the specific match ID's are then grabbed from  [opendota.com](https://www.opendota.com/) 
and installed into the directory of the user’s choice (which should be the location of their Dota 2 replays folder). The user can select 
the hero they want to download replays of and how many they want to download. After downloading is complete, a summary of what replays 
have been downloaded is shown. Duplicate replays (replays that have been requested to download but are already in the replays directory) 
are skipped and the user is prompted.

This program makes only one visit to [dota2protracker.com](https://www.dota2protracker.com/) per download request, regardless 
of the download amount, so the traffic to the website is just as minimal as if the one downloading was visiting the site and downloading 
replays personally. The same can be said for visits to [opendota.com](https://www.opendota.com/). Dota Pro Replay Downloader is just a tool 
that preforms the same actions one would take to download replays manually and should just be used as a time-saving replacement.

## Use
You will need Google Chrome installed to run this application.

Once opened, you should specify:
- What directory to download replays to, usually located in 
*Program Files (x86)\Steam\steamapps\common\dota 2 beta\game\dota\replays* 
- The hero which you want to download replays of
- The number of replays you wish to download

(Once a directory is selected it stays memorized. So one will only need to specify a directory once unless they want to change it in the future.)
Then simply press *Download* to install the replays

As a side note, one must still open the Dota 2 client and *download* the replays manually once they are in the replays directory.
Unfortunately, match replays cannot be fully installed through means outside of the client.

This does mean however, that space will not be taken up unnecessarily on your hard disk due to replays downloaded by the program, 
as only the file name of the replay need be correct to have the client recognize it as a downloadable replay.
In other words, until downloaded in-game, every replay downloaded through Dota Pro Replay Downloader is an empty file and takes up no space.

That being said, Valve does set a time limit on replay acquisition: replays are only available for **8 days** after the game takes place.
So if one wants to store a watchable replay, they should download it from the client before that time limit is reached.

## Installation
For installation, simply download and run the [setup.exe](setup.exe) file located in this repository.
To uninstall, find the download location and run the built-in uninstaller executable.

## Issues and Bugs
If you find a problem with this program or run into any crashes and/or bugs, please add an issue to let me know.

With every new Chrome version this program will have to be updated and repackaged with the corresponding chromedriver executeable. If
there are any crashes they are most likely due to the fact that I have not updated this yet. As stated before, if this is hindering your 
ability to download replays just leave a issue open. (Alternatively one could restore Chrome to the previous version as a workaround.)
