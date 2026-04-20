# -*- coding: utf-8 -*-
"""Test PyMOL headless mode fix"""
import os
import sys

# 设置headless环境变量
os.environ['PYMOL_HEADLESS'] = '1'
os.environ['PYMOL_QUIET'] = '1'

# 禁用OpenGL
os.environ['DISPLAY'] = ''

import pymol

print("Starting PyMOL in headless mode...")
try:
    # 不启动GUI
    pymol.finish_launching(['pymol', '-c'])  # -c = command line only
    print("[OK] PyMOL launched successfully")
    
    # 测试基本命令
    cmd = pymol.cmd
    cmd.reinitialize()
    cmd.pseudoatom('test')
    test_coords = cmd.get_coords('test')
    print('[OK] Basic command test successful')
    print('Test atom coordinates:', test_coords)
    
    # 清理
    cmd.delete('all')
    pymol.cmd.quit()
    print('[OK] PyMOL cleanup successful')
    
except Exception as e:
    print('[FAIL] PyMOL error:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n=== PyMOL环境测试通过 ===')
