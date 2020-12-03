
rem for %%f in (E:/ripcurrents/Holland/mp4/1215/*.mp4) do (
rem 	python contours.py --video E:/ripcurrents/Holland/mp4/1215/%%~nxf --out E:\ripcurrents\Holland\results --height 480 --window 1500
rem 	python contours_unit.py --video E:/ripcurrents/Holland/mp4/1215/%%~nxf --out E:\ripcurrents\Holland\results --height 480 --window 1500
rem 	python timex.py --video E:/ripcurrents/Holland/mp4/1215/%%~nxf --out E:\ripcurrents\Holland\results --height 480
rem )

rem for %%f in (E:/ripcurrents/Holland/mp4/mp4_all/1215/*.mp4) do (
rem 	python shear.py --video E:/ripcurrents/Holland/mp4/mp4_all/1215/%%~nxf --out E:/ripcurrents/Holland/mp4/results/shear_all/1215/%%~nxf --height 480 --window 1200
rem )


rem python shear.py --video E:\ripcurrents\Holland\mp4\selection\0829.mp4 --out E:/ripcurrents/Holland/mp4/shear_threshold/0829 --height 480 --window 1200
rem python shear.py --video E:\ripcurrents\Holland\mp4\selection\0904.mp4 --out E:/ripcurrents/Holland/mp4/shear_threshold/0904 --height 480 --window 1200
rem python shear.py --video E:\ripcurrents\Holland\mp4\selection\0917.mp4 --out E:/ripcurrents/Holland/mp4/shear_threshold/0917 --height 480 --window 1200
rem python shear.py --video E:\ripcurrents\Holland\mp4\selection\0918.mp4 --out E:/ripcurrents/Holland/mp4/shear_threshold/0918 --height 480 --window 1200
rem ::python shear.py --video E:\ripcurrents\Holland\mp4\selection\0926.mp4 --out E:/ripcurrents/Holland/mp4/shear_threshold/0926 --height 480 --window 1200
rem python shear.py --video E:\ripcurrents\Holland\mp4\selection\0929.mp4 --out E:/ripcurrents/Holland/mp4/shear_threshold/0929 --height 480 --window 1200
rem python shear.py --video E:\ripcurrents\Holland\mp4\selection\0917.mp4 --out E:/ripcurrents/Holland/mp4/shear_threshold/0917 --height 480 --window 1200
rem python shear.py --video E:\ripcurrents\Holland\mp4\selection\1016.mp4 --out E:/ripcurrents/Holland/mp4/shear_threshold/1016 --height 480 --window 1200
rem ::python shear.py --video E:\ripcurrents\Holland\mp4\selection\1020.mp4 --out E:/ripcurrents/Holland/mp4/shear_threshold/1020 --height 480 --window 1200


rem for %%f in (E:/ripcurrents/flow_paper/original_data/*.mp4) do (
rem 	python timex.py --video E:/ripcurrents/flow_paper/original_data/%%~nxf --out E:/ripcurrents/flow_paper/timex --height 480
rem )

for %%f in (E:/ripcurrents/flow_paper/original_data/*.mp4) do (
	python timelines.py --video E:/ripcurrents/flow_paper/original_data/%%~nxf --out E:/ripcurrents/flow_paper/timelines --height 480
)


python perspective_timelines.py --video E:/ripcurrents/flow_paper/original_data/other_rip_02.mp4 --out E:/ripcurrents/flow_paper --height 480


python perspective_timelines.py --video E:/ripcurrents/miami_short/miami_03_short.mp4 --out E:/ripcurrents/flow_paper --height 480

python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/other_data/other_rip_05.mp4 --out E:/ripcurrents/flow_paper --height 480


python test_lk.py --video E:/ripcurrents/flow_paper/other_data/other_rip_05.mp4 --out E:/ripcurrents/flow_paper --height 480