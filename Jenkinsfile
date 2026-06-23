pipeline {
  agent {
    kubernetes {
      yaml readFile('jenkins/k8s/agent-pod-template.yaml')
      defaultContainer 'python'
    }
  }

  parameters {
    string(name: 'ROUND', defaultValue: '1', description: 'Attack round number')
  }

  environment {
    PHISHING_PROJECT_PATH = '/workspace/phshing-email'
    RESULTS_DIR = 'results'
    HF_TOKEN = credentials('hf-token')
    GITHUB_TOKEN = credentials('github-token')
    GITHUB_REPO = credentials('github-repo')   // set as Jenkins secret text: 'owner/repo-name'
  }

  stages {

    stage('Download Artifacts') {
      steps {
        sh """
          mkdir -p results agent_workspace agent_traces
          python pipeline/download_artifact.py --round ${params.ROUND}
        """
      }
    }

    stage('Attack Tests') {
      parallel {
        stage('Injection') {
          steps {
            sh "python pipeline/run_attacks.py --samples results/round_${params.ROUND}_samples.json --detector injection --round ${params.ROUND}"
          }
        }
        stage('Jailbreak') {
          steps {
            sh "python pipeline/run_attacks.py --samples results/round_${params.ROUND}_samples.json --detector jailbreak --round ${params.ROUND}"
          }
        }
        stage('Insecure Output') {
          steps {
            sh "python pipeline/run_attacks.py --samples results/round_${params.ROUND}_samples.json --detector insecure_output --round ${params.ROUND}"
          }
        }
        stage('Indirect Injection') {
          steps {
            sh "python pipeline/run_attacks.py --samples results/round_${params.ROUND}_samples.json --detector indirect_injection --round ${params.ROUND}"
          }
        }
      }
    }

    stage('Merge + Score') {
      steps {
        sh "python pipeline/merge_results.py --round ${params.ROUND}"
      }
    }

    stage('Orchestrator Agent') {
      steps {
        sh "python agents/orchestrator_agent.py --round ${params.ROUND}"

        script {
          def decision = readJSON file: 'results/pipeline_decision.json'
          env.AI_ACTION    = decision.action
          env.ARGO_WORKFLOW = decision.argo_workflow
          env.SEVERITY     = decision.severity
          echo "AI Decision: action=${env.AI_ACTION}, workflow=${env.ARGO_WORKFLOW}, severity=${env.SEVERITY}"
        }
      }
      post {
        always {
          archiveArtifacts artifacts: 'results/pipeline_decision.json, agent_traces/**', allowEmptyArchive: true
        }
      }
    }

    stage('Quality Gate') {
      steps {
        sh "pytest tests/test_guardrail.py -v --tb=short"
      }
    }

    stage('Scorecard') {
      steps {
        sh "python reporting/scorecard.py --round ${params.ROUND} --output results/scorecard.html"
      }
    }

    stage('Adversarial Retrain') {
      when {
        expression {
          return env.AI_ACTION == 'retrain' || env.AI_ACTION == 'partial_retrain'
        }
      }
      steps {
        sh """
          python pipeline/retrain.py \
            --decision results/pipeline_decision.json \
            --round ${params.ROUND}
        """
        sh "pytest tests/test_retrain.py -v --tb=short"
      }
      post {
        success {
          archiveArtifacts artifacts: "models/v${params.ROUND}/**", allowEmptyArchive: true
        }
      }
    }

    stage('Trigger Argo') {
      steps {
        withCredentials([string(credentialsId: 'argo-token', variable: 'ARGO_TOKEN')]) {
          sh """
            python pipeline/trigger_argo.py \
              --workflow ${env.ARGO_WORKFLOW} \
              --round ${params.ROUND} \
              --model-path models/v${params.ROUND}/
          """
        }
      }
    }

  }

  post {
    always {
      publishHTML(target: [
        allowMissing: true,
        alwaysLinkToLastBuild: true,
        keepAll: true,
        reportDir: 'results',
        reportFiles: 'scorecard.html',
        reportName: 'Adversarial Guardrail Report'
      ])
    }
    failure {
      echo "Pipeline failed. Check agent traces in archived artifacts."
    }
  }
}
