import attrs

from .pyutils import create_logger


@attrs.define(kw_only = True, slots = True)
class GalaxyObject:
    name: str = attrs.field(validator = attrs.validators.instance_of(str), repr = True, init = True)
    redshift: float = attrs.field(validator = attrs.validators.instance_of(float), repr = lambda v: f"{v:.3f}", init = True)
    ra: float = attrs.field(validator = attrs.validators.instance_of(float), repr = lambda v: f"{v:.3f}", init = True)
    dec: float = attrs.field(validator = attrs.validators.instance_of(float), repr = lambda v: f"{v:.3f}", init = True)
    nh: float = attrs.field(validator = attrs.validators.instance_of(float), repr = True, init = True)
    r500_kpc: float = attrs.field(validator = attrs.validators.instance_of(float), repr = lambda v: f"{v:.1f} kpc", init = True)

    def __attrs_post_init__(self):
        logger = create_logger('GalaxyObject')
        logger.info(self)
