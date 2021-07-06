var ScatterChartModule  = function(series, canvas_width, canvas_height) {
    // Create the tag:
    var canvas_tag = "<canvas width='" + canvas_width + "' height='" + canvas_height + "' ";
    canvas_tag += "style='border:1px dotted'></canvas>";
    // Append it to #elements
    var canvas = $(canvas_tag)[0];
    $("#elements").append(canvas);
    // Create the context and the drawing controller:
    var context = canvas.getContext("2d");

    var chartData = {
        datasets: [{
        data: [],
            backgroundColor: '#2196f3'
        }]
    };

    var chartOptions = {
        responsive: true,
        tooltips: {
            mode: 'index',
            intersect: false
        },
        title: {
            display: true,
            text: 'Pool owner stake VS pool stake' //todo make configurable
        },
        hover: {
            mode: 'nearest',
            intersect: true
        },
        scales: {
          xAxes: [{
            type: 'linear',
            position: 'bottom',
            display: true,
            scaleLabel: {
                display: true,
                labelString: 'Pool owner stake'
            }
          }],
          yAxes: [{
              display: true,
              scaleLabel: {
                display: true,
                labelString: 'Pool stake'
              }
            }]
        },
        elements: {
            point:{
                radius: 5
            }
        },
        legend: {
            display: false
        }
  }

    var chart = new Chart(context, {
        type: 'scatter', //alternative bubble chart if I have 3 dimensions instead of 2: https://www.chartjs.org/docs/latest/charts/bubble.html
        data: chartData,
        options: chartOptions
    });

    this.render = function(data) {
        var pointData = [];

        for (i = 0; i < data.length; i+=2) {
            pointData.push({x: data[i], y: data[i+1]});
        }
        chart.data.datasets[0].data = pointData;
        console.log(chart.options)
        chart.update();
    };

    this.reset = function() {
        while (chart.data.labels.length) { chart.data.labels.pop(); }
        chart.data.datasets.forEach(function(dataset) {
            while (dataset.data.length) { dataset.data.pop(); }
        });
        chart.update();
    };
};
