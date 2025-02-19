import logging

from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Q, Subquery, OuterRef
from django.forms import IntegerField

from ..models.camera import Camera
from ..models.camera_alert import CameraAlert
from ..models.rule_version import RuleVersion

from ..serializers import camera_alert_serializer
from utils import exceptions as module_exceptions

logger = logging.getLogger('app')


def get_user_filter(_user):
    return Q()

def get_camera_alert_detail(camera_alert_id, user):
    try:
        user_filter = get_user_filter(user)

        # INSERT FIELD camera_name FROM camera INTO camera_alert
        qs = CameraAlert.objects.annotate(
            camera_name=Subquery(
                Camera.objects.filter(id=OuterRef('camera_id')).values('name')[:1]
            )
        )
        qs = qs.get(pk=camera_alert_id)
        logger.info("============== CAMERA ALERT QS:     ===========")
        logger.info(qs)

        # === DANG LAM DO 13/02/25===

        # STEP1: Lay rule_version_id tu RuleVersion dua tren: rule_id + version_number FROM TABLE CAMERA_ALERT
        rule_version_id = RuleVersion.objects.filter(
            rule_id = qs.rule_id,
            version_number = qs.version_number
        ).values_list('id', flat=True).first()

        logger.info("============== RULE VERSION ID:     ===========")
        logger.info(rule_version_id)

        # STEP2: Lay detail cua Version = rule_version_id FROM TABLE RULE_VERSION

        version_detail = RuleVersion.objects.get(pk=rule_version_id)
        logger.info("============== VERSION DETAIL:     =============")
        logger.info(version_detail)

        # STEP3: INSERT version_detail INTO camera_alert
        qs.version_detail = version_detail
        logger.info("============== SAU KHI NOI qs vs version_detail:     =============")
        logger.info(qs)

        return qs

    except ObjectDoesNotExist:
        raise module_exceptions.CameraAlertNotFound()


CameraOutputSerializerMaps = {
    'simple': camera_alert_serializer.CameraAlertFilterSerializer
}

def get_rule_version_id(rule_id, version_number):
    result_version_id = None
    qs = RuleVersion.objects.filter(rule_id = rule_id).filter(version_number = version_number)
    if not qs.exists():
        logger.info('get_rule_version_id FAILED !')
    else:
        logger.info('get_rule_version_id SUCCESS !')
        result_version_id = qs.get().id
    return result_version_id

def get_list_camera_alerts(validated_data, user):
    # filter
    my_filter = get_user_filter(user)

    # filter theo camera_id
    camera_id = validated_data.get('camera_id')
    if camera_id:
        my_filter &= Q(camera_id=camera_id)

    # filter theo name trong model Camera thong qua foreign key
    name = validated_data.get('name')
    if name:
        my_filter &= Q(camera__name__icontains = name)

    # return_mode
    return_mode = 'simple'


    # if return_mode == 'simple':
        # JOIN thu cong voi Subquery
        # qs = CameraAlert.objects.select_related('camera')

    qs = CameraAlert.objects.filter(my_filter).annotate(
        camera_name = Subquery(
            Camera.objects.filter(id=OuterRef('camera_id')).values('name')[:1]
        ),
        # version_name = Subquery(
        #     RuleVersion.objects.filter(id=OuterRef('version_id'))
        # )

    )
    logger.info("============= QS ==============")
    logger.info(qs)


    output_serializer_class = CameraOutputSerializerMaps.get(return_mode,
                                                             camera_alert_serializer.CameraAlertFilterSerializer)


    return qs, output_serializer_class

