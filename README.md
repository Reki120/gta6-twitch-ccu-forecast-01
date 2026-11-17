# GTA VI launch-day Twitch CCV forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Reki120/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Reki120/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (19 Nov 2026, UTC), given the first hour's print.

`main.py` (Python 3.13, standard library only) reads one case
(`/input/<id>.in`) and writes `/output/<id>.out`: for hours 2–24, five
quantiles (5%, 25%, 50%, 75%, 95%) of that hour's Twitch category CCV.

## Method

Each of the case's comparable past launches gives a release-day curve of
hourly Twitch category viewers, anchored at its hour-1 print. For hour `h`,
donors vote the log-ratio `ln(v_h / v_1)`, weighted by a Gaussian kernel on
how similar their hour-1 print is to the target's (bandwidth 0.9 in log
space); console-only launches get double weight (GTA VI is console-only).
Launches still pre-unlock in hour 1 (hour-1 print under 5% of the day's
peak) are left out. The median is the kernel-weighted median of the
log-ratios; the spread is the kernel-weighted IQR / 1.349, floored at
`0.15 * sqrt(h - 1)` (the house baseline's random-walk spread).

Leave-one-launch-out cross-validation on the kit's 29 rolling-midnight
comparables: pinball loss **0.266** on the log scale, vs **0.359** for the
house baseline (flat at hour 1, same spread floor).

## Entry

- Competition: [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
- License: MIT
