from django.urls import path
from django.shortcuts import redirect
from . import views

def root_redirect(request):
    if request.user.is_authenticated:
        return redirect('servers')
    return redirect('login')

urlpatterns = [
    path('', root_redirect, name='root'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('servers/', views.servers_view, name='servers'),
    path('models/', views.models_view, name='models'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('search/', views.search_view, name='search'),
    path('download/', views.download_model_view, name='download'),
    path('models/delete/', views.delete_model_view, name='delete_model'),
    path('start/', views.start_instance_view, name='start_instance'),
    path('stop/<int:instance_id>/', views.stop_instance_view, name='stop_instance'),
    path('delete/<int:instance_id>/', views.delete_instance_view, name='delete_instance'),
    path('logs/<str:model_name>/<int:port>/', views.logs_view, name='view_logs'),
    path('benchmark/', views.benchmark_view, name='benchmark'),
    path('benchmark/start/', views.start_benchmark_view, name='start_benchmark'),
    path('benchmark/<int:run_id>/', views.benchmark_detail_view, name='benchmark_detail'),
    path('benchmark/<int:run_id>/status/', views.benchmark_status_view, name='benchmark_status'),
]
