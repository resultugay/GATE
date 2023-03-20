#!/bin/bash


data=(
'career'
'nba'
'person'
'comm_'
)

timelinessAttr=(
"league_name,potential,player_positions,international_reputation"
"team_abbreviation,player_weight,pts"
"LN,status,kids"
"name,address,owner"
)

did=$1

python discovery_ccs.py ${data[${did}]} ${timelinessAttr[${did}]} './CCs_'${data[${did}]}'.txt' 2

