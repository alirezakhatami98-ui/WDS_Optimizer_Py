WDS Optimizer plugin directory (example)
========================================

This folder shows the format of a plugin.  To install a plugin:

  Windows:  copy *.py files to %USERPROFILE%\.wds_optimizer\plugins\
  Linux:    copy *.py files to ~/.wds_optimizer/plugins/
  macOS:    copy *.py files to ~/.wds_optimizer/plugins/

Or use the CLI:

  wds-optimize plugins load path/to/my_plugin.py

After copying, run:

  wds-optimize plugins list

to verify it loaded.  In the GUI, click "Reload plugins" in the
sidebar (v0.5.1).

WARNING: plugins are Python code with full system access.  Only
install plugins from sources you trust.  There is no sandbox.