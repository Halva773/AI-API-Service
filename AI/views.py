from django.shortcuts import render, HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib


model = joblib.load('AI/text_classification_model.pkl')


class PredictTextView(APIView):
    def post(self, request):
        # Получение данных из запроса
        text = request.data.get('text', '')
        if not text:
            return Response({'error': 'No text provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Применение модели для предсказания
        prediction = model.predict([text])
        
        # Возвращаем результат
        return Response({'prediction': prediction[0]/4}, status=status.HTTP_200_OK)
    

    def get(self, request):
        text = request.GET.get('text', '')
        
        if not text:
            return Response({'error': 'No text provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        prediction = model.predict([text])
        return Response({'prediction': prediction[0]/4}, status=status.HTTP_200_OK)