from rest_framework import serializers
from apps.referral.models import ReferralCode, ReferralReward

class ReferralCodeSerializer(serializers.ModelSerializer):
    user_email = serializers.CharField(source='user.email', read_only=True)

    class Meta:
        model = ReferralCode
        fields = ['id', 'user', 'user_email', 'code', 'expires_at', 'is_active', 'created_at']
        read_only_fields = ['user', 'code', 'created_at']

class ReferralRewardSerializer(serializers.ModelSerializer):
    referrer_email = serializers.CharField(source='referrer_user.email', read_only=True)
    referee_email = serializers.CharField(source='referee_user.email', read_only=True)

    class Meta:
        model = ReferralReward
        fields = ['id', 'referral_code', 'referrer_user', 'referrer_email', 'referee_user', 'referee_email', 'reward_type', 'value', 'description', 'status', 'earned_at', 'applied_at', 'created_at']
        read_only_fields = ['referral_code', 'referrer_user', 'referee_user', 'earned_at', 'applied_at', 'created_at']
