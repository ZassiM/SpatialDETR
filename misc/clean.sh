if [ -e "screenshots/" ];then rm -rfv "screenshots/*" ; fi 
if [ -e "video_gen/" ];then rm -r "video_gen/*" ; fi 
if [ -e "*egg-info" ];then rm -rf "*egg-info/" ; fi 
if [ -e "*__pycache__*" ];then rm -rf "*__pycache__*/" ; fi 
