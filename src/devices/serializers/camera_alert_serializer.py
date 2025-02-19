from rest_framework import serializers

from devices.models.camera_alert  import CameraAlert
from devices.models.rule_version import RuleVersion


class CameraAlertFilterSerializer(serializers.Serializer):

    id = serializers.IntegerField(required=False)

    camera_id = serializers.IntegerField(required=False)
    camera_name = serializers.CharField(required=False)
    rule_id = serializers.IntegerField(required=False)
    version_number = serializers.IntegerField(required=False)

    type = serializers.CharField(required=False)
    desc = serializers.CharField(required=False)


    class Meta:
        model = CameraAlert
        fields = ('id', 'camera_id','camera_name', 'rule_id', 'version_number', 'type', 'desc')

class VersionDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = RuleVersion
        fields = '__all__'

class CameraAlertDetailFilterSerializer(serializers.Serializer):

    id = serializers.IntegerField(required=False)

    camera_id = serializers.IntegerField(required=False)
    camera_name = serializers.CharField(required=False)
    rule_id = serializers.IntegerField(required=False)
    version_number = serializers.IntegerField(required=False)
    version_detail = VersionDetailSerializer(required=False)


    type = serializers.CharField(required=False)
    desc = serializers.CharField(required=False)

    class Meta:
        model = CameraAlert
        fields = ('id', 'camera_id','camera_name', 'rule_id', 'version_number', 'version_detail', 'type', 'desc')
