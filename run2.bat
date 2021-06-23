for %%f in (E:/ripcurrents/flow_paper/arrow_seg/wave_dir/*seg.avi) do (
	python export_frame.py --video E:/ripcurrents/flow_paper/arrow_seg/wave_dir/%%~nxf --out E:/ripcurrents/flow_paper/arrow_seg/wave_dir/img
)