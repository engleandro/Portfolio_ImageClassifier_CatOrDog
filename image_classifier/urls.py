from django.urls import path, include
from rest_framework.routers import SimpleRouter


app = 'image_classifier'

router = SimpleRouter() #router.register()

urlpatterns = [
    # APP ENDPOINTS
]
urlpatterns += router.urls

