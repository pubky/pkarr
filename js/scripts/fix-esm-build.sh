#!/bin/bash
cat >dist/esm/package.json <<!EOF
{
    "type": "module"
}
!EOF