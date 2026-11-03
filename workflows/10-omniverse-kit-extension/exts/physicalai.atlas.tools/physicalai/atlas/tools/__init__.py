"""physicalai.atlas.tools: Kit extension package.

Kit imports this module (``[[python.module]] name = "physicalai.atlas.tools"``) and instantiates every
``omni.ext.IExt`` subclass it finds. The import is guarded so the pure modules (``helpers``, ``usd_ops``) can be
imported and tested outside Kit.
"""

try:
    from .extension import AtlasToolsExtension as AtlasToolsExtension
except ImportError:  # not running inside Kit: omni.* is unavailable
    AtlasToolsExtension = None
