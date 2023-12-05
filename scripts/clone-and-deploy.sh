#!/bin/bash

# git clone
cd ..
sudo rm -rf cog-bark
git clone https://github.com/render-ai/cog-bark
bash cog-bark/scripts/deploy.sh