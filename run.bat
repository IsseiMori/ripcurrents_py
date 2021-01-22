
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


rem for %%f in (E:/ripcurrents/flow_paper/original_data/no/*.mp4) do (
rem 	python timex.py --video E:/ripcurrents/flow_paper/original_data/no/%%~nxf --out E:/ripcurrents/flow_paper/timex --height 480
rem )

rem for %%f in (E:/ripcurrents/flow_paper/original_data/*.mp4) do (
rem 	python timelines.py --video E:/ripcurrents/flow_paper/original_data/%%~nxf --out E:/ripcurrents/flow_paper/timelines --height 480
rem )

rem for %%f in (E:/ripcurrents/flow_paper/original_data/yes/*.mp4) do (
rem 	python color.py --video E:/ripcurrents/flow_paper/original_data/yes/%%~nxf --out E:/ripcurrents/flow_paper/colormaps/norm --height 480
rem )

rem for %%f in (E:/ripcurrents/flow_paper/original_data/no/*.mp4) do (
rem 	python color.py --video E:/ripcurrents/flow_paper/original_data/no/%%~nxf --out E:/ripcurrents/flow_paper/colormaps/norm --height 480
rem )

rem for %%f in (E:/ripcurrents/flow_paper/original_data/yes/*.mp4) do (
rem 	python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/yes/%%~nxf --out E:/ripcurrents/flow_paper/timelines2 --height 480
rem )

rem for %%f in (E:/ripcurrents/flow_paper/original_data/no/*.mp4) do (
rem 	python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/no/%%~nxf --out E:/ripcurrents/flow_paper/timelines2 --height 480
rem )


rem python perspective_timelines.py --video E:/ripcurrents/flow_paper/original_data/other_rip_02.mp4 --out E:/ripcurrents/flow_paper --height 480


rem python perspective_timelines.py --video E:/ripcurrents/miami_short/miami_03_short.mp4 --out E:/ripcurrents/flow_paper --height 480

rem python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/other_data/other_rip_05.mp4 --out E:/ripcurrents/flow_paper --height 480


rem python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/rip_15_stable.mp4 --out E:/ripcurrents/flow_paper --height 480 --correct_perspective 1

rem flowmap on no
rem timeline LK on all, 100,50% lines (final image, video) 

rem python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/yes/rip_15_stable.mp4 --out E:/ripcurrents/timelines2 --height 480


rem python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/yes/rip_03.mp4 --out E:/ripcurrents/flow_paper/timelines3 --height 480

rem python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/holland/1028_no.mp4 --out E:/ripcurrents/flow_paper/timelines_holland --height 480

rem for %%f in (E:/ripcurrents/flow_paper/original_data/holland/*.mp4) do (
rem 	python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/holland/%%~nxf --out E:/ripcurrents/flow_paper/timelines_holland --height 480
rem )

rem python timelines.py --video E:/ripcurrents/flow_paper/original_data/yes/other_rip_02.mp4 --out E:/ripcurrents/flow_paper --height 480

rem python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/yes/other_rip_02.mp4 --out E:/ripcurrents/flow_paper --height 480


rem for %%f in (E:/ripcurrents/other_data/rip.dec18.2020/*.mp4) do (
rem 	python color.py --video E:/ripcurrents/other_data/rip.dec18.2020/%%~nxf --out E:/ripcurrents/other_data/rip.dec18.2020/color --height 480 --window 900
rem )

rem for %%f in (E:/ripcurrents/other_data/rip.dec18.2020/*.mp4) do (
rem 	python perspective_timelinesLK.py --video E:/ripcurrents/other_data/rip.dec18.2020/%%~nxf --out E:/ripcurrents/other_data/rip.dec18.2020/timelines --height 480
rem )


rem python color.py --video E:/ripcurrents/other_data/rip.dec18.2020/20201218_154044.mp4 --out E:/ripcurrents/other_data/rip.dec18.2020 --height 480


for %%f in (E:/ripcurrents/other_data/jan01.21.gazos/*.mp4) do (
	python color.py --video E:/ripcurrents/other_data/jan01.21.gazos/%%~nxf --out E:/ripcurrents/other_data/jan01.21.gazos/color --height 480
)

for %%f in (E:/ripcurrents/other_data/jan01.21.gazos/*.mp4) do (
	python color_unit.py --video E:/ripcurrents/other_data/jan01.21.gazos/%%~nxf --out E:/ripcurrents/other_data/jan01.21.gazos/color_unit --height 480
)


rem for %%f in (E:/ripcurrents/other_data/jan01.21.gazos/*.mp4) do (
rem 	python perspective_timelinesLK.py --video E:/ripcurrents/other_data/jan01.21.gazos/%%~nxf --out E:/ripcurrents/other_data/jan01.21.gazos/timelines --height 480
rem )


python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper --height 480 --window 300 --grid 20 --bins 6