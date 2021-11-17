#!/usr/bin/env bash
sp_pid=`ps -ef | grep 'bentoml serve' | grep -v grep | awk '{print $2}'`

if [ -z "$sp_pid" ];
 then
  echo "[ not find sp-tomcat pid ]"
 else
  echo "find result: $sp_pid "
  kill -9 $sp_pid
 fi