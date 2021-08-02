/**
 *  Like mesa's chart module but with a few different options (title, axis labels and no legend)
 */

var MyChartModule = function(series, canvas_width, canvas_height) {
    // Create the tag:
    var canvas_tag = "<canvas width='" + canvas_width + "' height='" + canvas_height + "' ";
    canvas_tag += "style='border:1px dotted'></canvas>";
    // Append it to #elements
    var canvas = $(canvas_tag)[0];
    $("#elements").append(canvas);
    // Create the context and the drawing controller:
    var context = canvas.getContext("2d");

    var convertColorOpacity = function(hex) {

        if (hex.indexOf('#') != 0) {
            return 'rgba(0,0,0,0.1)';
        }

        hex = hex.replace('#', '');
        r = parseInt(hex.substring(0, 2), 16);
        g = parseInt(hex.substring(2, 4), 16);
        b = parseInt(hex.substring(4, 6), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',0.1)';
    };

    // Prep the chart properties and series:
    var datasets = []
    for (var i in series) {
        var s = series[i];
        var new_series = {
            label: s.title,
            xLabel: s.xLabel,
            yLabel: s.yLabel,
            tooltipText: s.tooltipText,
            borderColor: s.color,
            backgroundColor: convertColorOpacity(s.color),
            data: []
        };
        datasets.push(new_series);
    }

    var chartData = {
        labels: [],
        datasets: datasets
    };

    var animationComplete = false // used to skip downloading empty canvas at the beginning

    var chartOptions = {
        responsive: true,
        animation: {
            onComplete: function() {
                if (animationComplete) {
                    var a = document.createElement('a');
                    a.href = chart.toBase64Image();
                    a.download = 'line-chart.png';
                    a.click()
                    this.options.animation.onComplete = null; //disable after first render so that image is not downloaded upon hovering
                }
                else {
                    animationComplete = true
                }
            }
        },
        tooltips: {
            displayColors: false,
            mode: 'index',
            intersect: false,
            callbacks: {
                title: function (tooltipItems, data) {
                    return "Step " + tooltipItems[0].xLabel;
                },
                label: function(tooltipItem, data) {
                    var tooltipText = data.datasets[tooltipItem.datasetIndex].tooltipText;
                    return tooltipItem.yLabel.toFixed(4).replace(".0000", "") + tooltipText;
                }
            }
        },
        title: {
            display: true,
            text: datasets[0].label
        },
        hover: {
            mode: 'nearest',
            intersect: true
        },
        scales: {
            xAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: datasets[0].xLabel
                },
                ticks: {
                    maxTicksLimit: 11
                }
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: datasets[0].yLabel
                }
            }]
        },
        elements: {
            point:{
                radius: 0 //to hide points from line
            }
        },
        legend: {
            display: false
        }
    };

    var chart = new Chart(context, {
        type: 'line',
        data: chartData,
        options: chartOptions
    });

    this.render = function(data) {
        chart.data.labels.push(control.tick);
        for (i = 0; i < data.length; i++) {
            chart.data.datasets[i].data.push(data[i]);
        }
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
