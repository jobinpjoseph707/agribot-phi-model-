pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh 'pytest tests/'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying agrobot...'
                // Placeholder: Add deployment steps
            }
        }
    }
}
