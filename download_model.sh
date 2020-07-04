#!/bin/sh

curl -o './stored_models/model_weights.tar.xz' 'https://surfdrive.surf.nl/files/index.php/s/EuUqRp3VYLBmoBk/download'
tar -xf './stored_models/model_weights.tar.xz'
mv './290.model' './stored_models/model_weights.p'
rm './stored_models/model_weights.tar.xz'
