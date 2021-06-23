
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
rem timeline LK on all',' 100','50% lines (final image',' video) 

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


rem for %%f in (E:/ripcurrents/other_data/jan01.21.gazos/*.mp4) do (
rem 	python color.py --video E:/ripcurrents/other_data/jan01.21.gazos/%%~nxf --out E:/ripcurrents/other_data/jan01.21.gazos/color --height 480
rem )

rem for %%f in (E:/ripcurrents/other_data/jan01.21.gazos/*.mp4) do (
rem 	python color_unit.py --video E:/ripcurrents/other_data/jan01.21.gazos/%%~nxf --out E:/ripcurrents/other_data/jan01.21.gazos/color_unit --height 480
rem )


rem for %%f in (E:/ripcurrents/other_data/jan01.21.gazos/*.mp4) do (
rem 	python perspective_timelinesLK.py --video E:/ripcurrents/other_data/jan01.21.gazos/%%~nxf --out E:/ripcurrents/other_data/jan01.21.gazos/timelines --height 480
rem )


rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper --height 480 --window 300 --grid 20 --bins 6


rem for %%f in (E:/ripcurrents/flow_paper/original_data/dataset/*.mp4) do (
rem 	python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/%%~nxf --out E:/ripcurrents/flow_paper/multi_arrows2 --height 480 --window 600 --grid 20 --bins 6
rem )


rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 114 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 822 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_04.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 482 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_05.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 114 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_06.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 331 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_08.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 602 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_15.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 519 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_16.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 31 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_21.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 650 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_22.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 500 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 317 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_07.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 1600 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_08.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 1530 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_011.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 551 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_01.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 936 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_02.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 922 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_03.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 765 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_04.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 924 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_05.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 920 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_06.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 449 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_07.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 807 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_08.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 600 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_20.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 551 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_21.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 651 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_22.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 880 --grid 20 --bins 6


rem python export_frame.py --video E:/ripcurrents/flow_paper/multi_arrows/no_rip_01_vis_bin_hist.avi --out E:/ripcurrents/flow_paper/arrow_last_frame


rem for %%f in (E:/ripcurrents/flow_paper/multi_arrows/*.avi) do (
rem 	python export_frame.py --video E:/ripcurrents/flow_paper/multi_arrows/%%~nxf --out E:/ripcurrents/flow_paper/arrow_last_frame
rem )

rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_03.mp4 --out E:/ripcurrents/flow_paper/multi_arrows --height 480 --window 90 --grid 20 --bins 6

rem python color.py --video E:/ripcurrents/flow_paper/original_data/no/%%~nxf --out E:/ripcurrents/flow_paper/colormaps/norm --height 480

rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 720 --ripsize 28 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 720 --ripsize 21 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 720 --ripsize 14 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 720 --ripsize 10 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 720 --ripsize 7 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 720 --ripsize 5 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 720 --ripsize 3 --window 822

rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 480 --ripsize 28 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 480 --ripsize 21 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 480 --ripsize 14 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 480 --ripsize 10 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 480 --ripsize 7 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 480 --ripsize 5 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 480 --ripsize 3 --window 822

rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 360 --ripsize 28 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 360 --ripsize 21 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 360 --ripsize 14 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 360 --ripsize 10 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 360 --ripsize 7 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 360 --ripsize 5 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 360 --ripsize 3 --window 822

rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 180 --ripsize 28 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 180 --ripsize 21 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 180 --ripsize 14 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 180 --ripsize 10 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 180 --ripsize 7 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 180 --ripsize 5 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 180 --ripsize 3 --window 822

rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 90 --ripsize 28 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 90 --ripsize 21 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 90 --ripsize 14 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 90 --ripsize 10 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 90 --ripsize 7 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 90 --ripsize 5 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 90 --ripsize 3 --window 822


rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 300 --ripsize 28 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 300 --ripsize 21 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 300 --ripsize 14 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 300 --ripsize 10 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 300 --ripsize 7 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 300 --ripsize 5 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 300 --ripsize 3 --window 822

rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 250 --ripsize 28 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 250 --ripsize 21 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 250 --ripsize 14 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 250 --ripsize 10 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 250 --ripsize 7 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 250 --ripsize 5 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 250 --ripsize 3 --window 822

rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 200 --ripsize 28 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 200 --ripsize 21 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 200 --ripsize 14 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 200 --ripsize 10 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 200 --ripsize 7 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 200 --ripsize 5 --window 822
rem python color_exp.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/colormaps/ripsize --height 200 --ripsize 3 --window 822


rem for %%f in (E:/ripcurrents/flow_paper/colormaps/ripsize/*.avi) do (
rem 	python export_frame.py --video E:/ripcurrents/flow_paper/colormaps/ripsize/%%~nxf --out E:/ripcurrents/flow_paper/colormaps/ripsize/img
rem )

rem magick montage -label %f -frame 5 -background '#336699' -geometry +4+4 rip_02_180_28.jpg rip_02_720_21.jpg frame.jpg

rem magick mogrify -resize 160x90 -quality 100 -path resized *.jpg

rem magick montage -label %f -frame 5 -background '#336699' -tile 7x8 *.jpg frame.jpg


rem python export_frame.py --video E:/ripcurrents/flow_paper/colormaps/ripsize/rip_02_720_10.avi --out E:/ripcurrents/flow_paper/colormaps/ripsize --frame 271

rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 114 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 114 --grid 20 --bins 8
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 114 --grid 20 --bins 10
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 822 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 822 --grid 20 --bins 8
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 822 --grid 20 --bins 10
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 822 --grid 20 --bins 6
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 822 --grid 20 --bins 8
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 822 --grid 20 --bins 10


rem for %%f in (E:/ripcurrents/flow_paper/bin_nums/other_rip_02*.avi) do (
rem 	python export_frame.py --video E:/ripcurrents/flow_paper/bin_nums/%%~nxf --out E:/ripcurrents/flow_paper/bin_nums/img
rem )

rem magick montage -mode concatenate -label %f -frame 5 -background '#336699' -tile 4x9 *.jpg frame.jpg

rem magick mogrify -resize 426x240 -quality 100 -path resized *.jpg

rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/bin_nums --height 480 --window 114 --grid 20 --bins 10

rem python dyelines.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/dyelines --height 480

rem python color.py --video E:/ripcurrents/feb1621ws/20210216_130514.mp4 --out E:/ripcurrents/feb1621ws --height 720 --ripsize 10

rem python perspective_timelinesLK.py --video E:/ripcurrents/feb1621ws/stable/20210216_131027.mp4 --out E:/ripcurrents/feb1621ws/stable --height 480

rem python color.py --video E:/ripcurrents/feb1621ws/stable/20210216_130514.mp4 --out E:/ripcurrents/feb1621ws/stable --height 720 --window 300

rem magick montage -mode concatenate -background '#999999'  *.jpg frame.jpg

rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 114
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 822
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_04.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 482
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_05.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 114
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_06.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 331
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_08.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 602
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_15.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 519
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_16.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 31 
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_21.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 650
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_22.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 500
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 317 
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_07.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 1600 
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_08.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 1530 
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_011.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 551
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_01.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 936
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_02.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 922
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_03.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 765
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_04.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 924
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_05.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 920
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_06.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 449
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_07.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 807
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_08.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 600
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_20.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 551
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_21.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 651
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_22.mp4 --out E:/ripcurrents/flow_paper/ground_truth --frame 880

rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_04.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_05.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_06.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_08.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_15.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_16.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_21.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_22.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_07.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_08.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_011.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_01.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_02.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_03.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_04.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_05.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_06.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_07.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_08.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_20.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_21.mp4 --out E:/ripcurrents/flow_paper/ground_truth
rem python export_frame.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_22.mp4 --out E:/ripcurrents/flow_paper/ground_truth

rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/arrow_seg --height 480 --window 900 --grid 20

rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 114 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 822 --grid 20 --wave_dir 5
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_04.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 482 --grid 20 --wave_dir 5
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_05.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 114 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_06.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 331 --grid 20 --wave_dir 3
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_08.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 602 --grid 20 --wave_dir 3
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_15.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 519 --grid 20 --wave_dir 3
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_16.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 31 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_21.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 650 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_22.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 500 --grid 20 --wave_dir 3
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 317 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_07.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 1600 --grid 20 --wave_dir 5
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_08.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 1530 --grid 20 --wave_dir 3
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_11.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 551 --grid 20 --wave_dir 2
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 1000 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_21.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 900 --grid 20 --wave_dir 3
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_01.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 936 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_02.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 922 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_03.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 765 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_04.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 924 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_05.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 920 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_06.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 449 --grid 20 --wave_dir 3
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_07.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 807 --grid 20 --wave_dir 4
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_08.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 600 --grid 20 --wave_dir 5
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_20.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 551 --grid 20 --wave_dir 1
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_21.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 651 --grid 20 --wave_dir 1
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_22.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 880 --grid 20 --wave_dir 1

rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 114 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 822 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_04.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 482 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_05.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 114 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_06.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 331 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_08.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 602 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_15.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 519 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_16.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 31 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_21.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 650 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_22.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 500 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 317 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_07.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 1600 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_08.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 1530 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_11.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 551 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 1000 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_21.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 900 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_01.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 936 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_02.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 922 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_03.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 765 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_04.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 924 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_05.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 920 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_06.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 449 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_07.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 807 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_08.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 600 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_20.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 551 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_21.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 651 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_22.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 880 --grid 20

rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_08.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 602 --grid 20 --wave_dir 3
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/raw --height 720 --window 1000 --grid 20 --wave_dir 5
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 822 --grid 20
rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_07.mp4 --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir --height 720 --window 1600 --grid 20


rem python virtual_buoys.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out E:/ripcurrents/flow_paper --height 720

rem python color.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_07.mp4 --out E:/ripcurrents/flow_paper/original_data --height 720 --window 1



rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 900 --grid 20

rem python timex.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out E:/ripcurrents/flow_paper/new --height 480

rem python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_15.mp4 --out E:/ripcurrents/flow_paper/new --height 720

rem python grid_arrows_color.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 317 --grid 20

rem python color.py --video E:/ripcurrents/flow_paper/original_data/dataset/no_rip_07.mp4 --out E:/ripcurrents/flow_paper/original_data --height 720 --window 1

rem python color.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_06.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 331
rem python color.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_15.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 519

rem python perspective_timelinesLK.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_21.mp4 --out E:/ripcurrents/flow_paper/new --height 720

rem python grid_arrows.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 114
rem python grid_arrows_color.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 822
rem python grid_arrows_color.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_04.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 482

rem python grid_arrows_color.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 1000 --grid 20

rem python virtual_buoys.py --video E:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out E:/ripcurrents/flow_paper/new --height 720


rem python grid_arrows_mask.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 114
rem python grid_arrows_color_mask.py --video F:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out F:/ripcurrents/flow_paper/new --height 720 --window 114 --mask "F:/ripcurrents/flow_paper/figures/rip1/masks.png"
rem python grid_arrows_color_mask.py --video F:/ripcurrents/flow_paper/original_data/dataset/rip_02.mp4 --out F:/ripcurrents/flow_paper/new --height 720 --window 822 --mask "F:/ripcurrents/flow_paper/figures/rip2/masks.png"
rem python grid_arrows_color_mask.py --video F:/ripcurrents/flow_paper/original_data/dataset/rip_04.mp4 --out F:/ripcurrents/flow_paper/new --height 720 --window 482 --mask "F:/ripcurrents/flow_paper/figures/rip4/masks.png"
rem python grid_arrows_color_mask.py --video F:/ripcurrents/flow_paper/original_data/dataset/other_rip_20.mp4 --out F:/ripcurrents/flow_paper/new --height 720 --window 1000 --mask "F:/ripcurrents/flow_paper/figures/other20/masks.png"
rem python grid_arrows_color_mask.py --video F:/ripcurrents/flow_paper/original_data/dataset/no_rip_02.mp4 --out F:/ripcurrents/flow_paper/new --height 720 --window 922 --mask "F:/ripcurrents/flow_paper/figures/no2/masks.png"

rem python color.py --video E:/ripcurrents/flow_paper/original_data/dataset/rip_01.mp4 --out E:/ripcurrents/flow_paper/new --height 720 --window 1000
rem python color.py --video D:/Documents/Research/RipCurrents/ripcurrents_py/other_rip_20.mp4 --out D:/Documents/Research/RipCurrents/ripcurrents_py --height 720 --window 1000

python grid_arrows_color_mask.py --video F:/ripcurrents/flow_paper/original_data/dataset/other_rip_02.mp4 --out F:/ripcurrents/flow_paper/new --height 720 --window 317 --mask "F:/ripcurrents/flow_paper/figures/no2/masks.png"
